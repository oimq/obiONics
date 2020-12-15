default_configs = {
    'screen':{
        'canvas':{
            'width':600,
            'height':750,
        },
        'point':{
            'size':20
        },
        'line':{
            'width':3,
            'length':150
        },
        'world':{
            'pris':500
        }
    },
    'finger':{
        'number':5,
        'joint':3,
        'point':3
    }
}

from jSona import save

if __name__=="__main__" :
    save("./hands_default_configs.json", default_configs)