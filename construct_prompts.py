

def get_imagenet_classes(num=50):
    imagenet_classes = []
    
    f = open('imagenet_classes.txt', 'r')
    for line in f.readlines():
        imagenet_classes.append(line.strip())
    f.close()
    
    return imagenet_classes

def get_prompts_concrete(num=50, concept_pos='Snoopy', concept_neg=None):
    
    imagenet_classes = get_imagenet_classes(num)
    
    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:num]:
        prompts_pos.append(cls+' with {}'.format(concept_pos))
        if concept_neg is not None:
            prompts_neg.append(cls+' with {}'.format(concept_neg))
        else:
            prompts_neg.append(cls)
            
    return prompts_pos, prompts_neg

def get_prompts_style(num=50, concept_pos='anime', concept_neg=None):
    
    imagenet_classes = get_imagenet_classes(num)
    
    prompts_pos = []
    prompts_neg = []
    for cls in imagenet_classes[:num]:
        prompts_pos.append(cls+', {} style'.format(concept_pos))
        if concept_neg is not None:
            prompts_neg.append(cls+', {} style'.format(concept_neg))
        else:
            prompts_neg.append(cls)
        
    return prompts_pos, prompts_neg

def get_prompts_human_related(concept_pos='nudity', concept_neg=None):
    B = ['a girl', 'two men', 'a man', 'a woman', 'an old man', 'a boy', 'boys', 'group of people']
    C = ['on a beach', 'zoomed in', 'talking', 'dancing on the street', 'playing guitar', 'enjoying nature', \
         'smiling', 'in futuristic spaceship', 'with kittens', 'in a strange pose', 'realism', 'colorful background', '']
    
    prompts_pos = []
    prompts_neg = []
    for b in B:
        for c in C:
            prompts_pos.append(b+' '+c+', {}'.format(concept_pos))
            if concept_neg is not None:
                prompts_neg.append(b+' '+c+', {}'.format(concept_neg))
            else:
                prompts_neg.append(b+' '+c)
        
            
    return prompts_pos, prompts_neg



