import re, sys, collections

stopwords = set(open('stop_words.txt').read().split(','))
with open(sys.argv[1]) as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]
nwords = 0
n_non_stop_words = 0
max_words = max_chars = max_char_len = 0
i = 0
wf = {}
for line in lines:
    if (i % 1000) == 0:
        print(f'--- {i} ---') 
    ws = re.findall('[a-z]{2,}', line.lower())
    max_words = max(max_words, len(ws))
    max_chars = max(max_chars, len(line))
    if ws != []:
        x = len(max(ws, key=len))
        if x > max_char_len:
            max_char_len = x
    non_stop_words = [w for w in ws if w not in stopwords]
    counts = collections.Counter(non_stop_words)
    for w, c in counts.items():
        if w in wf:
            wf[w] = wf[w] + c
        else:
            wf[w] = c
    nwords = nwords + len(ws)
    n_non_stop_words = n_non_stop_words + len(non_stop_words)
    i = i + 1

nlines = len(lines)

print('Number of lines: ', nlines)
print('Total words: ', nwords)
print('Total non-stop words: ', len(non_stop_words))
print('Max words in a line: ', max_words)
print('Max chars in a line: ', max_chars)
print('Max chars length: ', max_char_len)
print('Unique words: ', len(wf))
print("Top 30:")
i = 0
sorted_wf = [(k, wf[k]) for k in sorted(wf, key=wf.get, reverse=True)]
for w, c in sorted_wf[:30]:
    print(w, '-', c)
    i = i + 1
