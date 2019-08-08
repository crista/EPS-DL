import re, sys, collections

stopwords = set(open('stop_words.txt').read().split(','))
with open(sys.argv[1]) as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]
words = []
max_words = max_chars = 0
i = 0
for line in lines:
    if (i % 1000) == 0:
        print(f'--- {i} ---') 
    ws = re.findall('[a-z]{2,}', line.lower())
    max_words = max(max_words, len(ws))
    max_chars = max(max_chars, len(line))
    words = words + ws
    i = i + 1

nlines = len(lines)
lines = []
non_stop_words = [w for w in words if w not in stopwords]
counts = collections.Counter(non_stop_words)

print('Number of lines: ', nlines)
print('Total words: ', len(words))
print('Total non-stop words: ', len(non_stop_words))
print('Max words in a line: ', max_words)
print('Max chars in a line: ', max_chars)
print('Unique words: ', len(set(words)))
print("Top 30:")
for (w, c) in counts.most_common(30):
    print(w, '-', c)