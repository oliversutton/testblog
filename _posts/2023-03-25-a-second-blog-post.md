---
layout: authored_post
title:  "A second blog post!"
date:   2023-03-25 07:46:38 +0000
categories: news
authors: 
  - Undreth Galgorady
---

Clicking the link on the author takes you to their personal profile page.

Blog posts can also contain maths $\mu = \frac{1}{k} \sum_{i=1}^{k} x_k $, and we can even number equations:

$$
\begin{align}
  f(x) + 1 &= 5,
  \\
  g(x) + f(x) &= -1.
  \notag
\end{align}
$$

We can put in[^citation] pictures:
![Lion]({{ '/images/people/lion.jpg' | relative_url }})

and embed videos from YouTube

<iframe width="560" height="315" src="https://www.youtube.com/embed/iQf77WWNt40" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

and `code` including in blocks with syntax highlighting
```python
def my_function(x):
  return x + 1

my_function(5) # returns 6
```
authors can choose the language of the syntax highlighting too, here's C++:
```cpp
int main() {
  std::vector<int> a;
  a.push_back(1);
  std::cout << a[0] << std::endl;
}
```

and a gif animation of cyclic competition:

![cyclic competition]({{ '/images/about/cycliccompetition.gif' | relative_url }})

We can even include interactive graphics generated in python, using the `mpld3` package

{% include_relative second-post-includes/figure.html %}

and an interactive animation:

{% include_relative second-post-includes/animation.html %}


### References
[^citation]: This is technically a footnote, but we can also use it as a citation. Clicking the arrow takes you back to the text.
