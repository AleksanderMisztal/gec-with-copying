import json
from copygec.mytokenizer import enc, ids_to_tokens

# def print_tokenized_sentence(sentence):
#   ids = enc(sentence)
#   tokens = ids_to_tokens(ids)
#   print(sentence)
#   print(" ".join(tokens))
#   print(ids[1:-1])

# def get_tok_counts(sentence):
#   ids = enc(sentence)
#   tokens = ids_to_tokens(ids)
#   s = tokens[1][0]
#   return sum([t[0] == s or not t[-1].isalpha() for t in tokens[1:-1]]), len(tokens)-2

# print_tokenized_sentence("Cats hate water .")
# print_tokenized_sentence("It is called Penrhyh Castle , and this building will tell you all about Wales .")
# print_tokenized_sentence("It must be a very interesting visit for the students who have never experienced an oriental atmosphere .")
# print_tokenized_sentence("haha, haha,ha, hahahah , haha. haha haha .")


# print(get_tok_counts("Cats hate water."))
# print(get_tok_counts("It is called Penrhyh Castle , and this building will tell you all about Wales ."))

# from copygec.dataloader import load_datasets

# data = load_datasets()['dev']

# data = [x for x,y in data]

# tt, tw = 0,0
# for x in data:
#   w, t = get_tok_counts(x)
#   tt += t
#   tw += w
# print('words per token:', round(tw / tt, 3))
# print('tokens per word:', round(tt / tw, 3))

print(ids_to_tokens(enc('correction')))
print(ids_to_tokens(enc('corrction')))