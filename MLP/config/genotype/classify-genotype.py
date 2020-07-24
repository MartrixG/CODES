classify = {
	"normal": {
		"1": ["0,group_dense_4_tanh"], 
		"2": ["1,group_dense_4_sigmoid", "0,dense_layer_sigmoid"], 
		"3": ["2,max_pool_3x3", "0,group_dense_4_relu"], 
		"4": ["1,group_dense_4_tanh", "0,none"], 
		"5": ["2,avg_pool_3x3", "4,dense_layer_tanh"], 
		"6": ["4,group_dense_4_tanh", "5,dense_layer_relu"], 
		"7": ["6,avg_pool_3x3", "2,group_dense_4_tanh"], 
		"8": ["7,dense_layer_sigmoid", "2,group_dense_4_sigmoid"]
	},
	"activation": ["tanh"]
}
