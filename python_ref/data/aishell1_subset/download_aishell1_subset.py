#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('modelscope/aishell1_subset', subset_name='default', split='train')
#您可按需配置 subset_name、split，参照“快速使用”示例代码p