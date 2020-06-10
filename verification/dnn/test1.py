import dataset

train_data, valid_data = dataset.xxxx('../data')
train_data, valid_data = dataset.get_dataset('../data', 'HAPT')
train_queue, valid_queue = dataset.get_data_loader(train_data, valid_data, 2)