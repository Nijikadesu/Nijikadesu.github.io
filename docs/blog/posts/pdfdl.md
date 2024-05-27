---
authors:
    - mingkun
categories:
    - 小技巧
date: 2024-05-27
tags:
    - tricks
slug:  download-pdf-online
---

# 如何下载网页内嵌的PDF

我在浏览老师上传到学习通的pdf时，发现有些pdf只支持在线浏览，这为课程学习带来了很多不便利，因此，在这篇博客中，我将分享我下载网页内嵌pdf的解决方案。

<!-- more -->

## Step 1
来到网页版学习通的课程页面，打开嵌入pdf的网页（我需要下载的在课程的章节栏）。

## Step 2
按F12打开控制台。

## Step 3
点击控制台上方菜单栏的`网络`选项。

## Step 4
在筛选器栏，仅勾选`Fetch/XHR`选项。

## Step 5
F5刷新页面，在下方列出的名称清单中，寻找包含`flag=normal`的选项，点击该名称来查看详细信息。

## Step 6
在菜单栏点击预览，此时应该可以看到pdf选项，并看到带双引号的pdf地址。

## Step 7
打开该地址指向的页面，便可以下载想要的pdf资源啦^^。