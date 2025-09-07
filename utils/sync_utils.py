import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import requests

from common.Logger import logger
from common.config import Config
from utils.file_manager import file_manager, checkpoint


class SyncUtils:
    """同步工具类，负责异步发送keys到外部应用"""

    def __init__(self):
        """初始化同步工具"""
        # Gemini Balancer 配置
        self.balancer_url = (
            Config.GEMINI_BALANCER_URL.rstrip("/") if Config.GEMINI_BALANCER_URL else ""
        )
        self.balancer_auth = Config.GEMINI_BALANCER_AUTH
        self.balancer_sync_enabled = Config.parse_bool(
            Config.GEMINI_BALANCER_SYNC_ENABLED
        )
        self.balancer_enabled = bool(
            self.balancer_url and self.balancer_auth and self.balancer_sync_enabled
        )

        # GPT Load Balancer 配置
        self.gpt_load_url = (
            Config.GPT_LOAD_URL.rstrip("/") if Config.GPT_LOAD_URL else ""
        )
        self.gpt_load_auth = Config.GPT_LOAD_AUTH
        # 解析多个group names (逗号分隔)
        self.gpt_load_group_names = (
            [
                name.strip()
                for name in Config.GPT_LOAD_GROUP_NAME.split(",")
                if name.strip()
            ]
            if Config.GPT_LOAD_GROUP_NAME
            else []
        )
        self.gpt_load_sync_enabled = Config.parse_bool(Config.GPT_LOAD_SYNC_ENABLED)
        self.gpt_load_enabled = bool(
            self.gpt_load_url
            and self.gpt_load_auth
            and self.gpt_load_group_names
            and self.gpt_load_sync_enabled
        )

        # 延迟初始化线程池
        self.executor = None
        self.saving_checkpoint = False

        # 周期性发送控制
        self.batch_interval = 60
        self.batch_timer = None
        self.shutdown_flag = False

        # GPT Load Balancer group ID 缓存 (15分钟缓存)
        self.group_id_cache: Dict[str, int] = {}
        self.group_id_cache_time: Dict[str, float] = {}
        self.group_id_cache_ttl = 15 * 60  # 15分钟

        if not self.balancer_enabled:
            logger.warning(
                "🚫 Gemini Balancer sync disabled - URL or AUTH not configured"
            )
        else:
            logger.info(f"🔗 Gemini Balancer enabled - URL: {self.balancer_url}")

        if not self.gpt_load_enabled:
            logger.warning(
                "🚫 GPT Load Balancer sync disabled - URL, AUTH, GROUP_NAME not configured or sync disabled"
            )
        else:
            logger.info(
                f"🔗 GPT Load Balancer enabled - URL: {self.gpt_load_url}, Groups: {', '.join(self.gpt_load_group_names)}"
            )

        # 延迟启动批量发送任务，仅在需要时才启动
        if self.balancer_enabled or self.gpt_load_enabled:
            self._ensure_executor()
            self._start_batch_sender()

    def _ensure_executor(self):
        """确保线程池已初始化"""
        if self.executor is None:
            try:
                self.executor = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="SyncUtils"
                )
                logger.info("✅ Thread pool initialized successfully")
            except RuntimeError as e:
                logger.error(f"❌ Failed to initialize thread pool: {e}")
                logger.warning("🔄 Falling back to synchronous mode")
                self.executor = None

    def add_keys_to_queue(self, keys: List[str]):
        """
        将keys同时添加到balancer和GPT load的发送队列

        Args:
            keys: API keys列表
        """
        if not keys:
            return

        # Acquire lock for checkpoint saving
        while self.saving_checkpoint:
            logger.info(
                f"📥 Checkpoint is currently being saved, waiting before adding {len(keys)} key(s) to queues..."
            )
            time.sleep(0.5)  # Small delay to prevent busy-waiting

        self.saving_checkpoint = True  # Acquire the lock
        try:
            # Gemini Balancer
            if self.balancer_enabled:
                initial_balancer_count = len(checkpoint.wait_send_balancer)
                checkpoint.wait_send_balancer.update(keys)
                new_balancer_count = len(checkpoint.wait_send_balancer)
                added_balancer_count = new_balancer_count - initial_balancer_count
                logger.info(
                    f"📥 Added {added_balancer_count} key(s) to gemini balancer queue (total: {new_balancer_count})"
                )
            else:
                logger.info(
                    f"🚫 Gemini Balancer disabled, skipping {len(keys)} key(s) for gemini balancer queue"
                )

            # GPT Load Balancer
            if self.gpt_load_enabled:
                initial_gpt_count = len(checkpoint.wait_send_gpt_load)
                checkpoint.wait_send_gpt_load.update(keys)
                new_gpt_count = len(checkpoint.wait_send_gpt_load)
                added_gpt_count = new_gpt_count - initial_gpt_count
                logger.info(
                    f"📥 Added {added_gpt_count} key(s) to GPT load balancer queue (total: {new_gpt_count})"
                )
            else:
                logger.info(
                    f"🚫 GPT Load Balancer disabled, skipping {len(keys)} key(s) for GPT load balancer queue"
                )

            file_manager.save_checkpoint(checkpoint)
        finally:
            self.saving_checkpoint = False  # Release the lock

    def _send_balancer_worker(self, keys: List[str]) -> str:
        """
        实际执行发送到balancer的工作函数（在后台线程中执行）

        Args:
            keys: API keys列表

        Returns:
            str: "ok" if success, otherwise an error code string.
        """
        try:
            logger.info(f"🔄 Sending {len(keys)} key(s) to balancer...")

            # 1. 获取当前配置
            config_url = f"{self.balancer_url}/api/config"
            headers = {
                "Cookie": f"auth_token={self.balancer_auth}",
                "User-Agent": "HajimiKing/1.0",
            }

            logger.info(f"📥 Fetching current config from: {config_url}")

            # 获取当前配置
            response = requests.get(config_url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"Failed to get config: HTTP {response.status_code} - {response.text}"
                )
                return "get_config_failed_not_200"

            # 解析配置
            config_data = response.json()

            # 2. 获取当前的API_KEYS数组
            current_api_keys = config_data.get("API_KEYS", [])

            # 3. 合并新keys（去重）
            existing_keys_set = set(current_api_keys)
            new_add_keys_set = set()
            for key in keys:
                if key not in existing_keys_set:
                    existing_keys_set.add(key)
                    new_add_keys_set.add(key)

            if len(new_add_keys_set) == 0:
                logger.info(f"ℹ️ All {len(keys)} key(s) already exist in balancer")
                # 不需要记录发送结果，因为没有实际发送新密钥
                return "ok"

            # 4. 更新配置中的API_KEYS
            config_data["API_KEYS"] = list(existing_keys_set)

            logger.info(
                f"📝 Updating gemini balancer config with {len(new_add_keys_set)} new key(s)..."
            )

            # 5. 发送更新后的配置到服务器
            update_headers = headers.copy()
            update_headers["Content-Type"] = "application/json"

            update_response = requests.put(
                config_url, headers=update_headers, json=config_data, timeout=60
            )

            if update_response.status_code != 200:
                logger.error(
                    f"Failed to update config: HTTP {update_response.status_code} - {update_response.text}"
                )
                return "update_config_failed_not_200"

            # 6. 验证是否添加成功
            updated_config = update_response.json()
            updated_api_keys = updated_config.get("API_KEYS", [])
            updated_keys_set = set(updated_api_keys)

            failed_to_add = [
                key for key in new_add_keys_set if key not in updated_keys_set
            ]

            if failed_to_add:
                logger.error(
                    f"❌ Failed to add {len(failed_to_add)} key(s): {[key[:10] + '...' for key in failed_to_add]}"
                )
                # 保存发送结果日志 - 部分成功的情况
                send_result = {}
                keys_to_log = []
                for key in new_add_keys_set:  # 只记录尝试新增的密钥
                    if key in failed_to_add:
                        send_result[key] = "update_failed"
                        keys_to_log.append(key)
                    else:
                        send_result[key] = "ok"
                        keys_to_log.append(key)
                if keys_to_log:  # 只有当有需要记录的密钥时才记录
                    file_manager.save_keys_send_result(keys_to_log, send_result)
                return "update_failed"

            logger.info(
                f"✅ All {len(new_add_keys_set)} new key(s) successfully added to balancer."
            )

            # 保存发送结果日志 - 只记录实际新增的密钥
            send_result = {key: "ok" for key in new_add_keys_set}
            if send_result:  # 只有当有新增密钥时才记录
                file_manager.save_keys_send_result(list(new_add_keys_set), send_result)

            return "ok"

        except requests.exceptions.Timeout:
            logger.error("❌ Request timeout when connecting to balancer")
            # 保存发送结果日志 - 所有密钥都失败
            send_result = {key: "timeout" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "timeout"
        except requests.exceptions.ConnectionError:
            logger.error("❌ Connection failed to balancer")
            # 保存发送结果日志 - 所有密钥都失败
            send_result = {key: "connection_error" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "connection_error"
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON response from balancer: {str(e)}")
            # 保存发送结果日志 - 所有密钥都失败
            send_result = {key: "json_decode_error" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "json_decode_error"
        except Exception as e:
            logger.error(f"❌ Failed to send keys to balancer: {str(e)}")
            traceback.print_exc()
            # 保存发送结果日志 - 所有密钥都失败
            send_result = {key: "exception" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "exception"

    def _get_gpt_load_group_id(self, group_name: str) -> Optional[int]:
        """
        获取GPT Load Balancer group ID，带缓存功能

        Args:
            group_name: 组名

        Returns:
            Optional[int]: 组ID，如果未找到则返回None
        """
        current_time = time.time()

        # 检查缓存是否有效
        if (
            group_name in self.group_id_cache
            and group_name in self.group_id_cache_time
            and current_time - self.group_id_cache_time[group_name]
            < self.group_id_cache_ttl
        ):
            logger.info(
                f"📋 Using cached group ID for '{group_name}': {self.group_id_cache[group_name]}"
            )
            return self.group_id_cache[group_name]

        # 缓存过期或不存在，重新获取
        try:
            groups_url = f"{self.gpt_load_url}/api/groups"
            headers = {
                "Authorization": f"Bearer {self.gpt_load_auth}",
                "User-Agent": "HajimiKing/1.0",
            }

            logger.info(f"📥 Fetching groups from: {groups_url}")

            response = requests.get(groups_url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.error(
                    f"Failed to get groups: HTTP {response.status_code} - {response.text}"
                )
                return None

            groups_data = response.json()

            if groups_data.get("code") != 0:
                logger.error(
                    f"Groups API returned error: {groups_data.get('message', 'Unknown error')}"
                )
                return None

            # 查找指定group的ID
            groups_list = groups_data.get("data", [])
            for group in groups_list:
                if group.get("name") == group_name:
                    group_id = group.get("id")
                    # 更新缓存
                    self.group_id_cache[group_name] = group_id
                    self.group_id_cache_time[group_name] = current_time
                    logger.info(
                        f"✅ Found and cached group '{group_name}' with ID: {group_id}"
                    )
                    return group_id

            logger.error(f"Group '{group_name}' not found in groups list")
            return None

        except Exception as e:
            logger.error(f"❌ Failed to get group ID for '{group_name}': {str(e)}")
            return None

    def _send_gpt_load_worker(self, keys: List[str]) -> str:
        """
        实际执行发送到GPT load balancer的工作函数（在后台线程中执行）

        Args:
            keys: API keys列表

        Returns:
            str: "ok" if success, otherwise an error code string.
        """
        try:
            logger.info(
                f"🔄 Sending {len(keys)} key(s) to GPT load balancer for {len(self.gpt_load_group_names)} group(s)..."
            )

            # 遍历所有group names，为每个group发送keys
            all_success = True
            failed_groups = []

            for group_name in self.gpt_load_group_names:
                logger.info(f"📝 Processing group: {group_name}")

                # 1. 获取group ID (使用缓存)
                group_id = self._get_gpt_load_group_id(group_name)

                if group_id is None:
                    logger.error(f"Failed to get group ID for '{group_name}'")
                    failed_groups.append(group_name)
                    all_success = False
                    continue

                # 2. 发送keys到指定group
                add_keys_url = f"{self.gpt_load_url}/api/keys/add-async"
                keys_text = ",".join(keys)

                add_headers = {
                    "Authorization": f"Bearer {self.gpt_load_auth}",
                    "Content-Type": "application/json",
                    "User-Agent": "HajimiKing/1.0",
                }

                payload = {"group_id": group_id, "keys_text": keys_text}

                logger.info(
                    f"📝 Adding {len(keys)} key(s) to group '{group_name}' (ID: {group_id})..."
                )

                try:
                    # 发送添加keys请求
                    add_response = requests.post(
                        add_keys_url, headers=add_headers, json=payload, timeout=60
                    )

                    if add_response.status_code != 200:
                        logger.error(
                            f"Failed to add keys to group '{group_name}': HTTP {add_response.status_code} - {add_response.text}"
                        )
                        failed_groups.append(group_name)
                        all_success = False
                        continue

                    # 解析添加keys响应
                    add_data = add_response.json()

                    if add_data.get("code") != 0:
                        logger.error(
                            f"Add keys API returned error for group '{group_name}': {add_data.get('message', 'Unknown error')}"
                        )
                        failed_groups.append(group_name)
                        all_success = False
                        continue

                    # 检查任务状态
                    task_data = add_data.get("data", {})
                    task_type = task_data.get("task_type")
                    is_running = task_data.get("is_running")
                    total = task_data.get("total", 0)
                    response_group_name = task_data.get("group_name")

                    logger.info(
                        f"✅ Keys addition task started successfully for group '{group_name}':"
                    )
                    logger.info(f"   Task Type: {task_type}")
                    logger.info(f"   Is Running: {is_running}")
                    logger.info(f"   Total Keys: {total}")
                    logger.info(f"   Group Name: {response_group_name}")

                except Exception as e:
                    logger.error(
                        f"❌ Exception when adding keys to group '{group_name}': {str(e)}"
                    )
                    failed_groups.append(group_name)
                    all_success = False
                    continue

            # 根据结果返回状态
            if all_success:
                logger.info(
                    f"✅ Successfully sent keys to all {len(self.gpt_load_group_names)} group(s)"
                )
                # 保存发送结果日志 - 所有密钥都成功
                send_result = {key: "ok" for key in keys}
                file_manager.save_keys_send_result(keys, send_result)
                return "ok"
            else:
                logger.error(
                    f"❌ Failed to send keys to {len(failed_groups)} group(s): {', '.join(failed_groups)}"
                )
                # 保存发送结果日志 - 部分或全部失败
                send_result = {
                    key: f"partial_failure_{len(failed_groups)}_groups" for key in keys
                }
                file_manager.save_keys_send_result(keys, send_result)
                return "partial_failure"

        except requests.exceptions.Timeout:
            logger.error("❌ Request timeout when connecting to GPT load balancer")
            send_result = {key: "timeout" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "timeout"
        except requests.exceptions.ConnectionError:
            logger.error("❌ Connection failed to GPT load balancer")
            send_result = {key: "connection_error" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "connection_error"
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON response from GPT load balancer: {str(e)}")
            send_result = {key: "json_decode_error" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "json_decode_error"
        except Exception as e:
            logger.error(
                f"❌ Failed to send keys to GPT load balancer: {str(e)}", exc_info=True
            )
            send_result = {key: "exception" for key in keys}
            file_manager.save_keys_send_result(keys, send_result)
            return "exception"

    def _start_batch_sender(self) -> None:
        """启动批量发送定时器"""
        if self.shutdown_flag:
            return

        # 检查executor是否可用
        if self.executor is None:
            logger.warning("🚫 Thread pool not available, skipping batch send task")
            return

        try:
            # 启动发送任务
            self.executor.submit(self._batch_send_worker)

            # 设置下一次发送定时器
            self.batch_timer = threading.Timer(
                self.batch_interval, self._start_batch_sender
            )
            self.batch_timer.daemon = True
            self.batch_timer.start()
        except RuntimeError as e:
            logger.error(f"❌ Failed to submit batch send task: {e}")
            logger.warning("🔄 Attempting synchronous fallback")

    def _batch_send_worker(self) -> None:
        """批量发送worker"""
        while self.saving_checkpoint:
            logger.info(
                "📥 Checkpoint is currently being saving, waiting before batch sending..."
            )
            time.sleep(0.5)

        self.saving_checkpoint = True
        try:
            # 加载checkpoint
            logger.info(
                f"📥 Starting batch sending, wait_send_balancer length: {len(checkpoint.wait_send_balancer)}, wait_send_gpt_load length: {len(checkpoint.wait_send_gpt_load)}"
            )
            # 发送gemini balancer队列
            if checkpoint.wait_send_balancer and self.balancer_enabled:
                balancer_keys = list(checkpoint.wait_send_balancer)
                logger.info(
                    f"🔄 Processing {len(balancer_keys)} key(s) from gemini balancer queue"
                )

                result_code = self._send_balancer_worker(balancer_keys)
                if result_code == "ok":
                    # 清空队列
                    checkpoint.wait_send_balancer.clear()
                    logger.info(
                        f"✅ Gemini balancer queue processed successfully, cleared {len(balancer_keys)} key(s)"
                    )
                else:
                    logger.error(
                        f"❌ Gemini balancer queue processing failed with code: {result_code}"
                    )

            # 发送gpt_load队列
            if checkpoint.wait_send_gpt_load and self.gpt_load_enabled:
                gpt_load_keys = list(checkpoint.wait_send_gpt_load)
                logger.info(
                    f"🔄 Processing {len(gpt_load_keys)} key(s) from GPT load balancer queue"
                )

                result_code = self._send_gpt_load_worker(gpt_load_keys)

                if result_code == "ok":
                    # 清空队列
                    checkpoint.wait_send_gpt_load.clear()
                    logger.info(
                        f"✅ GPT load balancer queue processed successfully, cleared {len(gpt_load_keys)} key(s)"
                    )
                else:
                    logger.error(
                        f"❌ GPT load balancer queue processing failed with code: {result_code}"
                    )

            # 保存checkpoint
            file_manager.save_checkpoint(checkpoint)
        except Exception as e:
            stacktrace = traceback.format_exc()
            logger.error(f"❌ Batch send worker error: {e}\n{stacktrace}")
            logger.error(f"❌ Batch send worker error: {e}")
        finally:
            self.saving_checkpoint = False  # Release the lock

    def shutdown(self) -> None:
        """关闭线程池和定时器"""
        self.shutdown_flag = True

        if self.batch_timer:
            self.batch_timer.cancel()

        if self.executor:
            self.executor.shutdown(wait=True)
        logger.info("🔚 SyncUtils shutdown complete")


# 创建全局实例
sync_utils = SyncUtils()
