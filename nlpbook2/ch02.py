
import warnings
from urllib3.exceptions import NotOpenSSLWarning
from fastai.text.all import *

path = untar_data(URLs.IMDB)
path.ls()

(path/'train').ls()

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')

dls.show_batch()

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

learn.fine_tune(4, 1e-2)

# learn.fine_tune(4, 1e-2)

learn.show_results()

learn.predict("That movie was wicked cool!")

# imdb = DataBlock(blocks=(TextBlock.from_folder(path), CategoryBlock),
#                  get_items=get_text_files,
#                  get_y=parent_label,
#                  splitter=GrandparentSplitter(valid_name='test'))
#
# dls = imdb.dataloaders(path)

dls_lm = TextDataLoaders.from_folder(path, is_lm=True, valid_pct=0.1)

dls_lm.show_batch(max_n=5)

learn = language_model_learner(
    dls_lm, AWD_LSTM, metrics=[accuracy, Perplexity()],
    path=path, wd=0.1).to_fp16()

learn.fit_one_cycle(1, 1e-2)

# learn.save('1epoch')
# learn = learn.load('1epoch')
# learn.unfreeze()

learn.fit_one_cycle(10, 1e-3)

# learn.save_encoder('finetuned')

TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2
preds = [learn.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]
print("\n".join(preds))

# dls_clas = TextDataLoaders.from_folder(
#     untar_data(URLs.IMDB), valid='test',
#     text_vocab=dls_lm.vocab)
#
# learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
#
# learn = learn.load_encoder('finetuned')
# learn.fit_one_cycle(1, 2e-2)
# learn.freeze_to(-2)
# learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))
# learn.freeze_to(-3)
# learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))
# learn.unfreeze()
# learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))
