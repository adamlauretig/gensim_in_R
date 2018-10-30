Using Gensim in R
================
Adam Lauretig
3/17/2018

Introduction
============

[Gensim](https://radimrehurek.com/gensim/) is a powerful Python library for text modeling. It incorporates a variety of models, many of which are not available in R. However, the recently developed [reticulate](https://cran.r-project.org/web/packages/reticulate/index.html) package provides a solution for the R user who wants to dip their toes in, without learning Python. It allows the user to call Python code which behaves like R code, and to seamlessly pass R and Python objects back and forth. In this document, I will show how to install `gensim`, call it from `reticulate`, estimate word embeddings, and perform vector arithmetic.

Setup
=====

I use Python 3.6, as distributed with [Anaconda](https://www.anaconda.com/download/#macos), and once Anaconda is installed, I install Gensim at the command line, using pip. To do this, I type

``` bash
pip install gensim
```

at the terminal.

I assume you are using recent versions of R, and RStudio. To install thereticulate package from CRAN:

``` r
install.packages("reticulate")
```

We'll also use the quanteda and stringr packages, to install them:

``` r
install.packages(c("quanteda", "stringr"))
```

Loading Gensim
==============

Importing Gensim with reticulate is very similar to loading an R package more generally:

``` r
library(reticulate)
gensim <- import("gensim") # import the gensim library
Word2Vec <- gensim$models$Word2Vec # Extract the Word2Vec model
multiprocessing <- import("multiprocessing") # For parallel processing
```

In gensim, we extract the `Word2Vec` object from the `models` object, using the `$` operator. Thanks to reticulate, object-oriented nature of python is changed into something R users can recognize, and we can treat `Word2vec` as we would any other R function

Prepping the data
=================

As an example model, we'll use the text of inauguration speeches from the `quanteda` package. we just want to extract the text we'll use to a character vector, which has 58 elements. We'll lowercase all of the tokens involved, remove punctuation, and then collapse the resulting `tokens` object back into a set of character vectors.

``` r
library(quanteda)
```

    ## Package version: 1.3.4

    ## Parallel computing: 2 of 8 threads used.

    ## See https://quanteda.io for tutorials and examples.

    ## 
    ## Attaching package: 'quanteda'

    ## The following object is masked from 'package:utils':
    ## 
    ##     View

``` r
library(stringr)
txt_to_use <- quanteda::data_corpus_inaugural$documents$texts
txt_to_use <- tolower(txt_to_use)
txt_to_use <- stringr::str_replace_all(txt_to_use, "[[:punct:]]", "")
txt_to_use <- stringr::str_replace_all(txt_to_use, "\n", " ")
txt_to_use <- (str_split(txt_to_use, " "))
```

Creating Word2vec
=================

In python, unlike `R`, we create the model we want to run *before* we run it, supplying it with the various parameters it will take. We'll create an object called `basemodel`, which uses the skip-gram w/negative sampling implementation of *word2vec*. We'll use a window size of 5, considering words within five words of each side of a target word. We'll do 3 sweeps through the data, but in practice, you should do more. We'll tell gensim to use skipgram "`sg`" with negative sampling "`ns`", rather than the hierarchical softmax. Finally, we'll use a dimensionality of 25, for the embedding dimensions, but again, in practice, you should probably use more.

``` r
basemodel = Word2Vec(
    workers = 1, # using 1 core
    window = 5L,
    iter = 3L, # iter = sweeps of SGD through the data; more is better
    sg = 1L,
    hs = 0L, negative = 1L, # we only have scoring for the hierarchical softmax setup
    size = 25L
)
```

Training the model
==================

To train the model, we'll first build a vocabulary from the inaugural speeches we cleaned earlier. We'll then call the `train` object from `basemodel`, the way you would call an object in `R`.

``` r
basemodel$build_vocab(sentences = txt_to_use)
basemodel$train(
  sentences = txt_to_use,
  epochs = basemodel$iter, 
  total_examples = basemodel$corpus_count)
```

    ## [1] 249606

Examining the Results
=====================

We can examine the output from the model, this will produce a vector 25 long. This, however, is not particularly informative.

``` r
basemodel$wv$word_vec("united")
```

    ##  [1] -0.133538678 -0.030018795 -0.233270764  0.124265224  0.303895473
    ##  [6]  0.074919127 -0.340074509  0.235601544  0.253037959 -0.071178004
    ## [11]  0.488432646 -0.088275813 -0.098585576 -0.170189530 -0.072561726
    ## [16]  0.164322332 -0.001201969  0.129144400 -0.095070623 -0.133255765
    ## [21] -0.333842367  0.089206211  0.140987307  0.327957392  0.046285406

But that isn't particularly informative. Instead, thanks to `reticulate`'s ability to communicate between `R` and python, we can bring the vectors into R, and then calculate cosine distance (a measure of word similarity).

``` r
library(Matrix)
embeds <- basemodel$wv$syn0
rownames(embeds) <- basemodel$wv$index2word


# function for cosine distance
closest_vector <- function(vec1, mat1){
  vec1 <- Matrix(vec1, nrow = 1, ncol = length(vec1))
  mat1 <- Matrix(mat1)
  mat_magnitudes <- rowSums(mat1^2)
  vec_magnitudes <- rowSums(vec1^2)
  sim <- (t(tcrossprod(vec1, mat1)/
      (sqrt(tcrossprod(vec_magnitudes, mat_magnitudes)))))
  sim2 <- matrix(sim, dimnames = list(rownames(sim)))
  
  w <- sim2[order(-sim2),,drop = FALSE]
  w[1:10,]
}
  
closest_vector(embeds["united", ], embeds)
```

    ##       united constitution           of      several        union 
    ##    1.0000000    0.9101438    0.9052755    0.8971754    0.8911366 
    ##     exercise    executive     congress           by          and 
    ##    0.8352804    0.8270193    0.8243935    0.8237670    0.8235068

``` r
closest_vector(embeds["united", ] + embeds["states", ], embeds)
```

    ## constitution       states       united           of        union 
    ##    0.9570964    0.9517907    0.9403348    0.9217996    0.9082334 
    ##      several    executive     exercise          the   government 
    ##    0.8656615    0.8524791    0.8278962    0.8263734    0.8195797

This result isn't bad, for such a small corpus, with relatively few vectors. We can even do more complicated vector arithmetic:

``` r
closest_vector(embeds["american", ] - embeds["war", ], embeds)
```

    ##   limitation      suppose      adverse       search       menace 
    ##    0.5016158    0.4317480    0.4310123    0.4248639    0.4248431 
    ## intelligence   coordinate   eighteenth      vicious        glory 
    ##    0.4178735    0.4123477    0.4117297    0.3961964    0.3841387

Conclusion
==========

Overall, this is an introduction to `reticulate`, and to estimating word embeddings with gensim. I showed how to prep text, estimate embeddings, and perform vector arithmetic on these embeddings.
