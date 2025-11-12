import json
import faiss
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import re

# ---------------------- 1. 依赖安装（首次运行前执行）----------------------
# 终端运行以下命令安装依赖：
# pip install faiss-cpu sentence-transformers numpy

# ---------------------- 2. 复用之前的JSON转文本函数 ----------------------
def json_to_text(recipe_json: Dict) -> str:
    """结构化JSON转自然语句文本（沿用之前的逻辑，优化了清洗）"""
    name = recipe_json.get('name', '未知名称')
    dish = recipe_json.get('dish', '家常菜')  # 替换Unknown为默认分类
    description = recipe_json.get('description', '无描述')
    
    # 食材处理
    ingredients = recipe_json.get('recipeIngredient', [])
    ingredients_str = '、'.join(ingredients) if ingredients else '无食材信息'
    
    # 步骤处理（优化标点，避免换行混乱）
    instructions = recipe_json.get('recipeInstructions', [])
    if instructions:
        instructions_str = '；'.join([f'{i+1}. {step.strip()}' for i, step in enumerate(instructions)])
    else:
        instructions_str = '无步骤信息'
    
    author = recipe_json.get('author', '未知作者')
    keywords = recipe_json.get('keywords', [])
    keywords_str = '、'.join(keywords) if keywords else '无关键词'
    
    # 组合文本（精简结构，提升检索相关性）
    text = (
        f"菜品：{name}（{dish}）。{description} "
        f"所需食材：{ingredients_str}。 "
        f"烹饪步骤：{instructions_str}。 "
        f"相关关键词：{keywords_str}。作者：{author}"
    )
    # 清洗文本（修正笔误、去除多余空格）
    text = text.replace('乱泉水', '矿泉水').strip().replace('  ', ' ')
    return text

# ---------------------- 3. 文本分割（Chunking）----------------------
def split_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """将单条食谱文本分割为合适的Chunk（适配Embedding模型）
    
    Args:
        text: 要分割的文本
        chunk_size: 每个chunk的目标大小（字符数）
        overlap: 相邻chunk之间的重叠大小（字符数）
    Returns:
        List[str]: 分割后的文本块列表
    """
    # 首先按句号分割为句子
    sentences = re.split('(。|；|，)', text)
    
    # 将分隔符放回句子中
    sentences = [''.join(i) for i in zip(sentences[::2], sentences[1::2] + [''])]
    sentences = [s for s in sentences if s.strip()]  # 移除空字符串
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # 如果当前chunk加上新句子长度在目标范围内，就添加到当前chunk
        if current_length + sentence_length <= chunk_size:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            # 如果当前chunk不为空，保存它
            if current_chunk:
                chunks.append(''.join(current_chunk))
            
            # 开始新的chunk，并包含部分重叠内容
            if chunks and overlap > 0:
                # 从上一个chunk的末尾取overlap大小的内容作为开始
                overlap_text = chunks[-1][-overlap:]
                current_chunk = [overlap_text, sentence]
                current_length = len(overlap_text) + sentence_length
            else:
                current_chunk = [sentence]
                current_length = sentence_length
    
    # 添加最后一个chunk（如果有的话）
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    # 过滤过短的Chunk（避免无意义片段）
    return [chunk for chunk in chunks if len(chunk) > 50]

# ---------------------- 4. Embedding生成（Sentence-BERT开源模型）----------------------
def init_embedding_model():
    """初始化Embedding模型（免费开源，中文效果好）"""
    # all-MiniLM-L6-v2：轻量、快速，适合入门；中文可替换为 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def generate_embeddings(model: SentenceTransformer, chunks: List[str]) -> np.ndarray:
    """将Chunk列表转为向量（Embedding）"""
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,  # 转为numpy数组，适配FAISS
        show_progress_bar=False
    )
    return embeddings.astype('float32')  # FAISS要求float32格式

# ---------------------- 5. FAISS向量库构建与存储 ----------------------
def build_faiss_index(embeddings: np.ndarray, chunks: List[str]) -> tuple[faiss.IndexFlatL2, List[str]]:
    """构建FAISS向量索引（快速相似性检索）"""
    # 初始化FAISS索引（L2距离：计算向量间欧式距离，越小越相似）
    dimension = embeddings.shape[1]  # Embedding维度（all-MiniLM-L6-v2为384维）
    index = faiss.IndexFlatL2(dimension)
    # 添加向量到索引
    index.add(embeddings)
    return index, chunks

def save_index_and_chunks(index: faiss.IndexFlatL2, chunks: List[str], save_path: str = 'recipe_rag'):
    """保存向量库和原始Chunk（后续检索需匹配原始文本）"""
    # 保存FAISS索引
    faiss.write_index(index, f'{save_path}_index.faiss')
    # 保存原始Chunk（与向量一一对应）
    with open(f'{save_path}_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"向量库已保存至：{save_path}_index.faiss 和 {save_path}_chunks.json")

# ---------------------- 6. 完整流程串联（批量处理数据集）----------------------
def full_pipeline(json_dataset: List[Dict]):
    """完整流程：JSON→文本→分割→Embedding→向量库"""
    # 步骤1：批量转换JSON为文本
    print("正在转换JSON为自然文本...")
    all_texts = [json_to_text(json_data) for json_data in json_dataset]
    
    # 步骤2：批量分割文本为Chunk
    print("正在分割文本...")
    all_chunks = []
    for text in all_texts:
        chunks = split_text(text)
        all_chunks.extend(chunks)
    print(f"共生成 {len(all_chunks)} 个有效Chunk")
    
    # 步骤3：初始化模型并生成Embedding
    print("正在初始化Embedding模型并生成向量...")
    model = init_embedding_model()
    embeddings = generate_embeddings(model, all_chunks)
    print(f"生成向量维度：{embeddings.shape}")  # 输出格式：(Chunk数量, 384)
    
    # 步骤4：构建并保存向量库
    print("正在构建FAISS向量库...")
    index, chunks = build_faiss_index(embeddings, all_chunks)
    save_index_and_chunks(index, chunks)
    
    print("✅ 全部流程完成！已准备好RAG检索所需的向量库和原始文本Chunk")
    return index, chunks, model

# ---------------------- 示例用法（直接运行即可）----------------------
if __name__ == "__main__":
    # 1. 你的数据集（替换为实际的JSON列表，可从文件读取）
    # 示例：从JSON文件读取数据集（如果你的数据存在文件中）
    # with open('your_dataset.json', 'r', encoding='utf-8') as f:
    #     json_dataset = json.load(f)
    
    # 这里用你提供的示例JSON作为演示（实际使用时替换为你的完整数据集）
    json_dataset = [
        {
            'name': '红烧滩羊肉',
            'dish': 'Unknown',
            'description': '每到桂花香满城后，就可以吃羊肉温补身体了',
            'recipeIngredient': [
                '1kg羊肉', '5片姜', '3瓣蒜', '适量花椒', '3勺老抽',
                '3片香叶', '2个八角', '1个干辣椒', '1块桂皮', '2勺料酒',
                '适量盐', '3根香菜', '2个小洋葱', '适量乱泉水'
            ],
            'recipeInstructions': [
                '滩羊肉在姜水里焯3分钟，姜水里加点料酒去腥',
                '再将羊肉反过来焯水2分钟',
                '把羊肉切2/3手掌大小，羊肉煮熟后会缩水，所以可以稍微大一点的',
                '热油里放入花椒、桂皮、香叶、生姜爆炒30秒，放入切好的羊肉翻炒3分钟左右，把羊肉里面的油煸炒出来，再加入老抽上色，把所有羊肉都上色后加入乱泉水，以漠过羊肉上面为准',
                '锅里水烧开后，换成砂锅中火慢炖40分钟',
                '分次吃',
                '超级软糯',
                '加点萝卜进去炖起来，解油腻'
            ],
            'author': 'author_25682',
            'keywords': [
                '红烧滩羊肉的做法', '红烧滩羊肉的家常做法',
                '红烧滩羊肉的详细做法', '红烧滩羊肉怎么做',
                '红烧滩羊肉的最正宗做法'
            ],
        },
        {"name": "西班牙金枪鱼沙拉", "dish": "金枪鱼沙拉", "description": "", "recipeIngredient": ["超市罐头装半盒金枪鱼(in spring water)", "2大片生菜", "5个圣女果", "半根黄瓜", "半个红柿椒", "半个紫洋葱", "1个七成熟水煮蛋", "适量红酒醋", "适量胡椒", "适量橄榄油"], "recipeInstructions": ["鸡蛋进水煮，七成熟捞出（依个人喜好），同时备其他菜", "生菜撕片，圣女果开半，黄瓜滚刀，红柿椒切丝，紫洋葱切丝，鸡蛋四均分", "金枪鱼去水", "撒黑胡椒，红酒醋和少许橄榄油", "拌匀，拍照，开动"], "author": "author_67696", "keywords": ["西班牙金枪鱼沙拉的做法", "西班牙金枪鱼沙拉的家常做法", "西班牙金枪鱼沙拉的详细做法", "西班牙金枪鱼沙拉怎么做", "西班牙金枪鱼沙拉的最正宗做法", "沙拉"]},
        {"name": "香草布里欧修吐司 少油少糖", "dish": "布里欧修", "description": "6月初做了一大瓶香草糖，刚好休假吐司可以做起来啦😁做个卡路里爆炸的布里欧修，刚好消耗下家里剩下的材料。参考了几个布里欧修的方子，根据自己的喜好调整了下，成品口感松软香甜超好吃哦(º﹃º )\n已经减过油和糖啦，再减就不是布里欧休的口感啦(✪▽✪)鸡蛋蓬松柔软的蛋香味，搭着香草的香气，加上淡奶油黄油奶粉的奶香味，真的不错哦(//∇//)\n\n450克吐司两条\n#042", "recipeIngredient": ["【烫种】", "50克高筋面粉", "5克糖", "5克盐", "80克开水", "【中种】", "250克高筋面粉", "10克香草糖", "2克酵母", "40克淡奶油", "120克+-10克牛奶", "【主面团】", "250克高筋面粉", "25克奶粉", "50克，喜甜加量+10～30克香草糖", "2.5克盐", "6克酵母", "2个，约100克鸡蛋", "60克+-10克牛奶", "40-60克黄油/椰子油", "适量，可不加葡萄干"], "recipeInstructions": ["【香草糖】\n500克优质白砂糖+5根香草荚\n香草荚剖开，刮出香草籽，切成小段，和白砂糖混合在一起，摇匀，密封保存就做好啦。\n各种甜点都能用，自己做真的是物美价廉，使用也超方便，遍布香草籽，做出来的甜品高级感又提升不少。\n用的是宜家买的大号玻璃密封罐\n\n没有香草糖的同学可以用香草精代替哦", "【中种】\n所有材料混合，走一个揉面程序，密封发酵1-2小时至2倍大，冷藏24-48小时后使用\n\n【烫种】\n粉类混合后，称量开水80克，立即浇入粉中，快速搅拌均匀，密封冷藏24-48小时后使用", "除黄油外其他所有主面团材料+烫种+撕成小块的中种全部混合后，走一个揉面程序至粗膜状态，加入油类后再来一个揉面程序，揉至手套膜状态。\n团圆，密封发酵1小时左右，至两倍大", "取出轻拍排气，称量后均分成6份，分别团圆，盖上保鲜膜醒发10分钟", "擀卷一次，继续醒发10分钟", "擀卷第二次时卷入泡发后吸干水分的葡萄干，排入吐司盒，发酵至8-9分满，预热烤箱175度", "中间剪开，挤上黄油，175度烤30-35分钟，烤5分钟后盖锡纸哦(´-ω-`)", "长高高，长胖胖，就是锡纸盖晚了上色深了点", "来看下截面，松软拉丝香气十足"], "author": "author_1765", "keywords": ["香草布里欧修吐司 少油少糖的做法", "香草布里欧修吐司 少油少糖的家常做法", "香草布里欧修吐司 少油少糖的详细做法", "香草布里欧修吐司 少油少糖怎么做", "香草布里欧修吐司 少油少糖的最正宗做法", "面包", "吐司", "烤箱", "烘焙"]},
        {"name": "可可咖啡曲奇", "dish": "咖啡曲奇", "description": "", "recipeIngredient": ["100g黄油", "65g糖粉", "100g常温淡奶油25℃左右(冬天需温热到45℃）", "120g低筋面粉", "16g可可粉", "4g速溶咖啡粉", "50g玉米淀粉", "10g杏仁粉", "若干巧克力豆"], "recipeInstructions": ["黄油室温软化，加入糖粉搅打均匀", "分三次加入淡奶油，搅打至淡黄色的羽霜状，不可完全打发", "把低粉，可可粉，咖啡粉，玉米淀粉，杏仁粉混合后过筛加入（2）中，用刮刀翻拌+压扮均匀。 接着把面糊装入配好裱花嘴的裱花袋中，在烤盘上挤出大小相同的曲奇", "放入预热好的烤箱内烤20分钟\n我的温度是上火180℃,下火170℃，最后几分钟开热风循环一下\n时间和温度大家酌情增减哈"], "author": "author_64761", "keywords": ["可可咖啡曲奇的做法", "可可咖啡曲奇的家常做法", "可可咖啡曲奇的详细做法", "可可咖啡曲奇怎么做", "可可咖啡曲奇的最正宗做法", "曲奇"]}

        # 可在此处添加更多JSON数据（你的完整数据集）
    ]
    
    # 2. 执行完整流程
    index, chunks, embedding_model = full_pipeline(json_dataset)
    
    # ---------------------- 测试检索（验证效果）----------------------
    print("\n📌 测试检索功能：")
    user_query = "告诉我怎么做西班牙金枪鱼沙拉！"  # 用户问题
    # 问题转向量
    query_embedding = generate_embeddings(embedding_model, [user_query])
    # 检索Top-2最相似的Chunk
    k = 2
    distances, indices = index.search(query_embedding, k)
    
    # 输出检索结果
    print(f"用户问题：{user_query}")
    print("检索到的相关内容：")
    for i in range(k):
        idx = indices[0][i]
        print(f"\n第{i+1}条相关内容（相似度：{1 - distances[0][i]/2:.2f}）：")
        print(chunks[idx])