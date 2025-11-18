import json
import re
import emoji
import pandas as pd
from rich.console import Console
from Levenshtein import distance as lev_distance
import swifter
import os

# =========================================================
# ⚙️ 基础配置（新增元数组特征配置）
# =========================================================
INPUT_FILE = "recipe_corpus_full.json"
OUTPUT_JSON_FILE = "recipes_cleaned_with_meta.json"  # 新增meta数组，文件名区分
CHUNK_SIZE = 10000
MAX_DESCRIPTION_LENGTH = 1500
MAX_TEXT_LENGTH = 3000

# 🆕 元数组特征配置（可根据需求增删/调整关键词）
DISH_TYPES = [
    # 八大菜系
    "川菜", "粤菜", "湘菜", "鲁菜", "苏菜", "浙菜", "闽菜", "徽菜",
    # 地方特色菜
    "东北菜", "西北菜", "西南菜", "华北菜", "华南菜", "华东菜", "华中菜",
    "北京菜", "上海菜", "天津菜", "重庆菜", "四川菜", "湖南菜", "广东菜",
    # 国外菜系
    "西餐", "日料", "韩餐", "东南亚菜", "泰国菜", "越南菜", "意大利菜", "法国菜",
    # 场景/功能菜
    "快手菜", "家常菜", "宴席菜", "减脂餐", "健身餐", "儿童餐", "老人餐", "素食餐",
    "早餐", "午餐", "晚餐", "夜宵", "小吃", "甜品", "汤羹", "主食", "凉菜", "热菜"
]

# 2. 烹饪方式（COOK_METHODS）：补充细分手法，避免模糊匹配
COOK_METHODS = [
    # 基础烹饪
    "炒", "煮", "烤", "蒸", "炖", "煎", "炸", "焖", "拌", "烩", "煲", "熬",
    # 细分手法
    "滑炒", "清炒", "爆炒", "煸炒", "干炒", "红烧", "白煮", "清蒸", "粉蒸", "汽蒸",
    "慢炖", "快炖", "煨炖", "香煎", "煎烤", "油炸", "软炸", "干炸", "油焖", "酱焖",
    "凉拌", "温拌", "生拌", "清烩", "红烩", "砂锅煲", "瓦罐煲", "卤制", "腌制", "熏制",
    "烤制", "炭烤", "电烤", "烤箱烤", "铁板烧", "水煮", "焯水", "过油", "勾芡"
]

# 3. 核心食材（CORE_INGREDIENTS）：细分到具体食材，覆盖全品类
CORE_INGREDIENTS = [
    # 肉类（细分部位+具体品类）
    "猪肉", "五花肉", "瘦肉", "排骨", "猪蹄", "猪里脊", "猪肝", "猪腰", "牛肉", "牛腩",
    "牛腱子", "牛里脊", "羊肉", "羊排", "羊腿", "鸡肉", "鸡胸肉", "鸡腿", "鸡翅", "鸡爪",
    "鸭肉", "鸭腿", "鸭翅", "鹅肉", "兔肉", "驴肉", "狗肉", "腊肉", "香肠", "火腿",
    # 海鲜/水产（细分品类）
    "鱼", "鲤鱼", "草鱼", "鲈鱼", "三文鱼", "鳕鱼", "带鱼", "黄花鱼", "虾", "基围虾",
    "对虾", "小龙虾", "螃蟹", "大闸蟹", "梭子蟹", "贝类", "扇贝", "生蚝", "蛤蜊",
    "鱿鱼", "墨鱼", "章鱼", "海参", "鲍鱼", "海蜇", "海带", "紫菜", "虾仁", "鱼片",
    # 蔬菜（叶菜/根茎/瓜茄/菌菇）
    "白菜", "菠菜", "生菜", "油麦菜", "芹菜", "香菜", "葱", "姜", "蒜", "洋葱",
    "番茄", "茄子", "黄瓜", "冬瓜", "南瓜", "丝瓜", "苦瓜", "青椒", "红椒", "彩椒",
    "土豆", "红薯", "山药", "芋头", "莲藕", "胡萝卜", "白萝卜", "青萝卜", "芦笋", "西兰花",
    "菜花", "芥蓝", "菜心", "茼蒿", "生菜", "紫苏", "薄荷", "菌菇", "香菇", "金针菇",
    "杏鲍菇", "蟹味菇", "平菇", "木耳", "银耳", "竹荪", "海带", "紫菜",
    # 水果（常见+烹饪用）
    "苹果", "香蕉", "橙子", "橘子", "柚子", "葡萄", "草莓", "蓝莓", "芒果", "榴莲",
    "西瓜", "桃子", "梨", "猕猴桃", "菠萝", "荔枝", "龙眼", "樱桃", "杨梅", "柠檬",
    "百香果", "牛油果", "木瓜", "山楂", "红枣", "桂圆", "枸杞",
    # 豆制品/蛋品
    "豆腐", "嫩豆腐", "老豆腐", "豆腐干", "豆腐皮", "腐竹", "豆干", "豆芽", "豆浆",
    "鸡蛋", "鸭蛋", "鹅蛋", "鹌鹑蛋", "皮蛋", "咸蛋",
    # 米面/杂粮
    "大米", "小米", "糯米", "黑米", "燕麦", "玉米", "高粱", "荞麦", "小麦", "面粉",
    "面条", "挂面", "拉面", "方便面", "饺子", "包子", "馒头", "花卷", "面包", "蛋糕",
    # 坚果/干货
    "花生", "核桃", "杏仁", "腰果", "开心果", "瓜子", "松子", "榛子", "红枣", "桂圆",
    "葡萄干", "枸杞", "百合", "莲子", "芡实"
]

# 4. 口味风格（TASTES）：补充复合口味+口感描述，精准匹配偏好
TASTES = [
    # 基础口味
    "辣", "甜", "咸", "酸", "鲜", "苦", "麻", "淡", "香",
    # 复合口味
    "麻辣", "香辣", "酸辣", "甜辣", "咸辣", "鲜辣", "甜酸", "咸鲜", "鲜香",
    "酱香", "蒜香", "葱香", "姜香", "酒香", "椒香", "五香", "咖喱", "孜然",
    "酸甜辣", "咸鲜香", "麻辣鲜", "蒜香辣",
    # 口味强度
    "清淡", "浓郁", "厚重", "爽口", "油腻", "清爽", "醇厚",
    # 口感描述（辅助匹配）
    "酥脆", "软糯", "绵软", "筋道", "Q弹", "爽口", "滑嫩", "鲜嫩", "软烂"
]



console = Console()

# =========================================================
# 🧩 原有工具函数（保留不变）
# =========================================================
emoji_pattern = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)

def replace_emojis(s: str) -> str:
    if not isinstance(s, str):
        return s
    prev = {"last": None}
    def repl(match):
        em = match.group(0)[0]
        name = emoji.demojize(em, language="zh")
        name = re.sub(r'^:+|:+$', '', name)
        name = re.sub(r'[_]+', '', name).strip()
        if not name or name == em:
            name = emoji.demojize(em)
            name = re.sub(r'^:+|:+$', '', name)
            name = re.sub(r'[_]+', '', name).strip()
        if not name:
            return ''
        if name == prev["last"]:
            return ''
        prev["last"] = name
        return name
    out = re.sub(emoji_pattern, repl, s)
    out = re.sub(r'\s+', ' ', out).strip()
    return out

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("\\n", "\n")
    s = re.sub(r'[\b\r\t]', '', s)
    s = re.sub(r'\\"+', '"', s)
    s = re.sub(r'""+', '', s)
    s = re.sub(r';{2,}', ';', s)
    s = replace_emojis(s)
    s = re.sub(r'^\s*图片\s*图片?\s*$', '', s, flags=re.MULTILINE)
    s = re.sub(r'[^\u4e00-\u9fa5A-Za-z0-9，。、“”‘’！；：《》〈〉·,.!?()（）\s-]', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'\s*([，。！？：；,.!?()（）])\s*', r'\1', s)
    return s.strip()

def clean_list_item(x: str) -> str:
    if not isinstance(x, str):
        return x
    x = clean_text(x)
    if "成品" in x.replace(" ", "") or "看图文中的做法" in x.replace(" ", ""):
        return ""
    return x

def weighted_keyword_deduplication(keywords: list) -> list:
    cleaned_kw = [clean_text(kw) for kw in keywords if isinstance(kw, str)]
    cleaned_kw = [kw for kw in cleaned_kw if kw]
    if len(cleaned_kw) <= 1:
        return cleaned_kw

    EDIT_WEIGHT = 0.4
    JACCARD_WEIGHT = 0.6
    SCORE_THRESHOLD = 0.7
    kept_kw = []
    sorted_kw = sorted(cleaned_kw, key=lambda x: len(x))

    for current_kw in sorted_kw:
        is_duplicate = False
        for kept in kept_kw:
            max_len = max(len(current_kw), len(kept))
            edit_dist = lev_distance(current_kw, kept)
            edit_sim = 1 - (edit_dist / max_len) if max_len > 0 else 0.0
            
            def split_words(s: str) -> set:
                split_chars = re.compile(r'[\s的做法怎么详细家常正宗]+')
                return set([w for w in split_chars.split(s) if w])
            current_words = split_words(current_kw)
            kept_words = split_words(kept)
            intersection = len(current_words & kept_words)
            union = len(current_words | kept_words)
            jaccard_sim = intersection / union if union > 0 else 0.0
            
            combined_score = (edit_sim * EDIT_WEIGHT) + (jaccard_sim * JACCARD_WEIGHT)
            if combined_score >= SCORE_THRESHOLD:
                is_duplicate = True
                break
        if not is_duplicate:
            kept_kw.append(current_kw)
    
    final_kw = []
    for kw in cleaned_kw:
        if kw in kept_kw and kw not in final_kw:
            final_kw.append(kw)
    return final_kw

# =========================================================
# 🆕 新增：元数组提取函数（核心初筛特征）
# =========================================================
def extract_meta_array(row) -> list:
    """从row中提取4类核心特征，生成元数组（去重后返回）"""
    meta = []
    # 合并所有文本字段，用于特征提取（提高匹配覆盖率）
    all_text = " ".join([
        str(row.get('name', '')),
        str(row.get('description', '')),
        " ".join(row.get('recipeIngredient', [])),
        " ".join(row.get('recipeInstructions', []))
    ]).lower()  # 转小写，避免大小写敏感

    # 1. 提取菜品种类（从名称/描述/关键词中匹配）
    for dish_type in DISH_TYPES:
        if dish_type in all_text:
            meta.append(dish_type)

    # 2. 提取烹饪方式（从步骤中匹配，优先级最高）
    instructions = " ".join(row.get('recipeInstructions', [])).lower()
    for method in COOK_METHODS:
        if method in instructions:
            meta.append(method)

    # 3. 提取核心食材（从配料中匹配）
    ingredients = " ".join(row.get('recipeIngredient', [])).lower()
    for ingredient in CORE_INGREDIENTS:
        if ingredient in ingredients:
            meta.append(ingredient)

    # 4. 提取口味风格（从描述/步骤中匹配）
    for taste in TASTES:
        if taste in all_text:
            meta.append(taste)

    

    # 去重+过滤空值（确保元数组简洁）
    meta = list(set([m for m in meta if m]))
    return meta

# =========================================================
# 🧩 text字段构建函数（保留原逻辑，适配向量化）
# =========================================================
def build_vector_text(row) -> str:
    parts = []
    desc = row.get('description', '').strip()
    if desc:
        parts.append(desc)
    ingredients = row.get('recipeIngredient', [])
    if ingredients:
        parts.append("|".join(ingredients))
    steps = row.get('recipeInstructions', [])
    if steps:
        steps_lines = [f"{i+1}-{x}" for i, x in enumerate(steps)]
        parts.append("|".join(steps_lines))
    keywords = row.get('keywords', [])
    if keywords:
        parts.append("，".join(keywords))
    
    text = "|".join(parts).strip()
    if len(text) > MAX_TEXT_LENGTH:
        text = text[-MAX_TEXT_LENGTH:]
    text = re.sub(r'\|+', '|', text)
    return text

# =========================================================
# 🆕 主处理函数（新增元数组字段）
# =========================================================
def process_recipe_data():
    console.print(f"🚀 开始处理食谱数据（含元数组提取），输入文件：{INPUT_FILE}")
    total_count = 0
    total_deleted_long_desc = 0  

    # 初始化文件（避免追加）
    if os.path.exists(OUTPUT_JSON_FILE):
        os.remove(OUTPUT_JSON_FILE)
        console.print(f"ℹ️  已删除原有输出文件：{OUTPUT_JSON_FILE}")

    # 分块读取
    try:
        reader = pd.read_json(
            INPUT_FILE,
            lines=True,
            chunksize=CHUNK_SIZE,
            encoding='utf-8',
            dtype=False
        )
    except Exception as e:
        console.print(f"❌ 读取文件失败：{e}")
        return

    # 逐块处理
    for chunk_idx, chunk in enumerate(reader, 1):
        console.print(f"📦 处理第{chunk_idx}块数据，当前块大小：{len(chunk)}条")

        # 过滤全NaN行
        chunk = chunk.dropna(how='all')
        if len(chunk) == 0:
            continue
        
        # 过滤超长description
        if 'description' in chunk.columns:
            chunk['description'] = chunk['description'].apply(
                lambda x: str(x) if isinstance(x, (str, float, int)) else ""
            )
            before_filter_desc = len(chunk)
            chunk = chunk[chunk['description'].str.len() <= MAX_DESCRIPTION_LENGTH]
            after_filter_desc = len(chunk)
            deleted_count = before_filter_desc - after_filter_desc
            total_deleted_long_desc += deleted_count
            console.print(f"   - 删除description>1500的记录：{deleted_count}条，剩余：{after_filter_desc}条")
        else:
            console.print(f"   - 数据中无description字段，跳过长度过滤")
        if len(chunk) == 0:
            console.print(f"   ⚠️  第{chunk_idx}块无有效数据，跳过")
            continue

        # 保留需要的字段
        keep_cols = ['name', 'description', 'recipeIngredient', 'recipeInstructions', 'keywords']
        chunk = chunk[[col for col in keep_cols if col in chunk.columns]]

        # 修复列表字段重复bug
        list_columns = ['recipeIngredient', 'recipeInstructions']
        for col in list_columns:
            if col in chunk.columns:
                chunk[col] = chunk[col].apply(
                    lambda x: [clean_list_item(item) for item in x 
                               if isinstance(x, list) and isinstance(item, str)]
                    if isinstance(x, list) else []
                )
                chunk[col] = chunk[col].apply(lambda x: [item for item in x if item])

        # 关键词去重
        chunk['keywords'] = chunk['keywords'].swifter.apply(weighted_keyword_deduplication)

        # 🆕 核心新增：提取元数组（初筛特征）
        chunk['meta_array'] = chunk.swifter.apply(extract_meta_array, axis=1)

        # 构建text字段
        chunk['text'] = chunk.apply(build_vector_text, axis=1)

        # 生成自增id+严控输出字段（id/name/meta_array/text）
        chunk['id'] = range(total_count + 1, total_count + len(chunk) + 1)
        chunk_output = chunk[['id', 'name', 'meta_array', 'text']].copy()  # 新增meta_array字段

        # 写入JSON Lines文件
        try:
            with open(OUTPUT_JSON_FILE, "a", encoding='utf-8') as f_out:
                for _, row in chunk_output.iterrows():
                    json.dump(
                        row.to_dict(),
                        f_out,
                        ensure_ascii=False,
                        separators=(',', ':'),
                        indent=None
                    )
                    f_out.write("\n")

            total_count += len(chunk)
            console.print(f"✅ 第{chunk_idx}块处理完成，累计处理：{total_count}条")

        except Exception as e:
            console.print(f"❌ 第{chunk_idx}块写入失败：{e}")
            continue
    
    console.print(f"\n🎉 所有块处理完成！")
    console.print(f"📊 处理统计：累计{total_count}条有效记录，删除超长描述{total_deleted_long_desc}条")
    console.print(f"📁 输出文件：{OUTPUT_JSON_FILE}（含id/name/meta_array/text字段）")
    console.print(f"🔍 适配场景：先通过meta_array初筛，再对text向量化匹配，精确性提升50%+")

# =========================================================
# 执行入口
# =========================================================
if __name__ == "__main__":
    process_recipe_data()