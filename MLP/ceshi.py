import json

classify = {'classify': {}}
classify['classify']['normal'] = {}
classify['classify']['activation'] = {}
classify['classify']['normal']['1'] = ["0,dense_layer_relu"]
classify['classify']['normal']['2'] = ["1,group_dense_4_tanh", "0,group_dense_4_sigmoid"]
classify['classify']['normal']['3'] = ["0,dense_layer_relu"]
classify['classify']['normal']['4'] = ["1,group_dense_4_tanh", "0,group_dense_4_sigmoid"]
classify['classify']['normal']['5'] = ["0,dense_layer_relu"]
classify['classify']['normal']['6'] = ["1,group_dense_4_tanh", "0,group_dense_4_sigmoid"]
classify['classify']['activation']['1'] = 'tanh'
classify['classify']['activation']['2'] = 'sigmoid'
classify['classify']['activation']['3'] = 'tanh'
classify['classify']['activation']['4'] = 'sigmoid'
with open('test.json', 'w') as f:
    json.dump(classify, f, indent=4)
with open('test.json', 'r') as f:
    test = json.load(f)
print(test)