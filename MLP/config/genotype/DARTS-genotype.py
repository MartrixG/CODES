DARTS_V2 = {
    "normal": {
        "2": ["0,sep_conv_3x3", "1,sep_conv_3x3"],
        "3": ["0,sep_conv_3x3", "1,sep_conv_3x3"],
        "4": ["1,sep_conv_3x3", "0,skip_connect"],
        "5": ["0,skip_connect", "2,dil_conv_3x3"]
    },
    "reduce": {
        "2": ["0,max_pool_3x3", "1,max_pool_3x3"],
        "3": ["2,skip_connect", "1,max_pool_3x3"],
        "4": ["2,skip_connect", "2,skip_connect"],
        "5": ["1,max_pool_3x3", "1,max_pool_3x3"]
    }
}