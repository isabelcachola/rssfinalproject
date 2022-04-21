# rssfinalproject

## Requirements

This code has been tested with python 3.8.  
To install required packages, run: 
`pip install -r requirements.txt`

## Usage

The first time the code is run, pretrained translation models and required files will be downloaded.
This may take a few seconds to minutes depending on the speed of your internet connection. 

To translate a sentence between Spanish and English, you can run for example:
```
$ python models.py --lang_pair en-es --src "this is a test"
Esto es una prueba.
INFO - Time to run script: 4.008340358734131 secs
```

To run an individual swap, identify the (0-indexed) token you would like to swap with `swap_idx` and the value you would like to swap in with `swap_val`:
```
$ python analysis.py --lang_pair en-es --src "this is a test" --swap_idx 3 --swap_val sentence
cossim between src (test) and sub (sentence) is: 0.297076.
cossim between output (prueba.) and sub (frase.) is: 0.905399.
Esto es una prueba.
Esta es una frase.
INFO - Time to run script: 13.224300146102905 secs
```

To run top-N analysis, you will need the precomputed cosine similarity matrices for the language pair you are interested in.
These are large (14G) and so not committed here, but should be put in the `precomputed_cos_sim` dir. 
The tarred files are located on the clsp grid at: 
```
/export/b02/icachola/rssfinalproject/precomputed_cos_sims/{en/es}.tar.gz
```
and expanded files at:
```
/export/c24/salesky/rssfinalproject/precomputed_cos_sims/{en/es}/
```

For top-n analysis, we again target a specific (0-indexed) token (`swap_idx`) in the source sentence (`--src "__"`) to be translated. 
Top-N analysis can be computed using an integer N (`swap_n`) or a percentage N (`swap_percent`)
```
$ python topn_analysis.py --lang_pair en-es --src "this is a sentence" --swap_idx 3 --swap_n 3
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.58it/s]
  swap_token_idx     swap_val cos_input cos_output  cos_diff               src_swap                tgt_swap
  1          39426  ▁sentencing  0.989939   0.949038  0.040901  this is a ▁sentencing  Esto es una sentencia.
  0          54005    ▁Sentence   0.98929   0.949038  0.040252    this is a ▁Sentence  Esto es una sentencia.
  2          14442   ▁sentences  0.994717   0.991565  0.003151   this is a ▁sentences      Esto es una frase.

$ python topn_analysis.py --lang_pair en-es --src "this is a king" --swap_idx 3 --swap_n 3
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:01<00:00,  2.24it/s]
  swap_token_idx swap_val cos_input cos_output  cos_diff          src_swap              tgt_swap
  0          21477   ▁royal  0.986014   0.909036  0.076978  this is a ▁royal  Esto es una realeza.
  1          19125   ▁queen  0.987222   0.955796  0.031426  this is a ▁queen    Esta es una reina.
  2          20245   ▁kings  0.990785   0.991114   0.00033  this is a ▁kings       Esto es un rey.

$ python topn_analysis.py --lang_pair en-es --src "this is a king" --swap_idx 3 --swap_percent 0.001
Swapping 0.001% yields 66 swaps
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 66/66 [00:36<00:00,  1.79it/s]
   swap_token_idx      swap_val cos_input cos_output  cos_diff                src_swap                tgt_swap
   0           49699     ▁monarchy  0.983002    0.88902  0.093982     this is a ▁monarchy  Esto es una monarquía.
   10          16728  ▁politicians  0.982777   0.889965  0.092812  this is a ▁politicians    Este es un político.
   63          49498        ▁spies  0.983917   0.894607   0.08931        this is a ▁spies       Esto es un espía.
   38          18373        ▁reign  0.983947   0.894999  0.088947        this is a ▁reign     Esto es un reinado.
   28          36258      ▁princes  0.984852    0.89626  0.088592      this is a ▁princes    Este es un príncipe.
   32          26784       ▁prince  0.983032    0.89626  0.086772       this is a ▁prince    Este es un príncipe.
   27          21477        ▁royal  0.986014   0.909036  0.076978        this is a ▁royal    Esto es una realeza.
   52          52259      ▁royalty  0.984458   0.909036  0.075421      this is a ▁royalty    Esto es una realeza.
   40          29906       ▁rulers  0.984202   0.911976  0.072226       this is a ▁rulers  Esto es un gobernante.
   6           53584     ▁monastic  0.983691   0.914838  0.068852     this is a ▁monastic   Este es un monástico.
   INFO - Time to run script: 0.7990408460299174 mins
```

Additional (non-commandline) functionality such as word alignment, etc is demonstrated in `init_testing.ipynb`
