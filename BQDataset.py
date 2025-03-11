from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('C-MTEB/BQ', subset_name='default', split='train')

print(ds[10])