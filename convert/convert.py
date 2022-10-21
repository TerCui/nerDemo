import json

def get_token(input):
    english = 'abcdefghijklmnopqrstuvwxyz'
    output=[]
    buffer=''
    for s in input:
        if s in english or s in english.upper():
            buffer+=s
        else:
            if buffer:output.append(buffer)
            buffer=''
            output.append(s)
    if buffer: output.append(buffer)
    return output

def json2bio(fpath,output):
    with open(fpath) as f:
        lines = f.readlines()
        for line in lines:
            annotions = json.loads(line)
            # print(annotions)
            text = annotions['text'].replace('\n',' ')
            # print(text)
            all_words = get_token(text.replace(' ',','))
            all_labels = ['O']*len(all_words)
            for i in annotions['label']:
                b_location = i[0]
                e_location = i[1]
                label = i[2]
                all_labels[b_location] = 'B-'+label
                if b_location!=e_location:
                    for word in range(b_location+1,e_location):
                        all_labels[word] = 'I-'+label
            cur_line =0
            token_label = zip(all_words,all_labels)
            with open(output,'a',encoding='utf-8') as f:
                for tl in token_label:
                    f.write(tl[0]+' '+tl[1])
                    f.write('\n')
                    cur_line+=1
                    if cur_line == len(all_words):
                        f.write('\n')


if __name__ == '__main__':
    fpath = 'admin.jsonl'
    output = 'tain.txt'
    json2bio(fpath,output)