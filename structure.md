---
layout: post
title: Structure
permalink: /structure/
---


[text](jekyll/update/2024/03/31/Fine-tune-copy.html)

jekyll serve --future
jekyll serve --trace

# username.github.io/  (or your custom repo for GitHub Pages)
│
├── _config.yml      (Site configuration)
├── Gemfile          (Gem dependencies for local testing)
│
├── _posts/          (Blog posts)
│   ├── 2024-03-29-welcome-to-my-blog.md
│   └── 2024-04-01-my-second-post.md
│
├── _pages/          (Contains non-post pages)
│   ├── about.md     (About Me page)
│   ├── cv.md        (Curriculum Vitae)
│   ├── projects.md  (Projects page)
│   └── contact.md   (Contact page)
│
├── _layouts/        (Custom layouts)
│   ├── default.html (The default layout)
│   ├── post.html    (Layout for blog posts)
│   └── page.html    (Layout for regular pages)
│
├── _includes/       (Reusable components, e.g., header, footer)
│   ├── header.html
│   ├── footer.html
│   └── navbar.html
│
├── assets/
│   ├── css/         (CSS files)
│   │   └── main.scss
│   ├── img/         (Image files)
│   └── js/          (JavaScript files)
│
├── _data/           (Data files, like navigation.yml for site navigation)
│
├── index.md         (The Home page content)
├── blog.md          (Optional, if you want a main Blog page linking to posts)
│
└── _site/           (Automatically generated site output, not checked in)
