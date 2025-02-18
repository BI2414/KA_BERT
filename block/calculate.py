import yake
from transformers import BertTokenizer, BertModel
# from src.utils import ChunkExample

# 假设使用 YAKE 算法来提取关键词
def PTM_keyword_extractor_yake(text):
    # 使用 YAKE 提取关键词
    yake_extractor = yake.KeywordExtractor(lan="en", n=3)  # n=3 表示提取最多 3 个词的关键词
    keywords = yake_extractor.extract_keywords(text)

    # 返回提取到的关键词（根据得分排序）
    return [keyword[0] for keyword in keywords]  # 提取关键词，不包括分数


# # 示例文本
# text = "Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human languages."
#
# sentence1 = "A tropical storm rapidly developed in the Gulf of Mexico Sunday and was expected to hit somewhere along the Texas or Louisiana coasts by Monday night .	"
# sentence2 = "A tropical storm rapidly developed in the Gulf of Mexico on Sunday and could have hurricane-force winds when it hits land somewhere along the Louisiana coast Monday night ."
# # 提取关键词
# keywords1 = PTM_keyword_extractor_yake(sentence1)
# keywords2 = PTM_keyword_extractor_yake(sentence2)
# print("Extracted Keywords1:", keywords1)
# print("Extracted Keywords2:", keywords2)