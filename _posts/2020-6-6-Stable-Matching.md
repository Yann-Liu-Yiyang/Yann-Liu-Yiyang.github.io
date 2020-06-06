---
layout: post
title:  "Stable Matching Problem"
date:   2020-6-6 14:09:00 +0800
tags: algorithm
color: rgb(123,45,45)
cover: '../assets/yuan.jpg'
subtitle: 'stable matching'
---

# Stable Matching Problem

## 1. Example 

Matching Residents to Hospital

+ The goal of the problem: Given a set of preferences among hospital and medical school students, design a self-reinforcing admissions process.

+ Unstable pair: applicant x and hospital y are unstable if:
  + x prefers y to its assigned hospital.
  + y prefers x to one of its admitted students.
+ Stable assignment.  Assignment with no unstable pairs.
  + Natural and desirable condition
  + Individual self-interest will prevent any applicant/hospital deal from being made.

## 2. Problem setting

+ Goal: Given n men and n women, find a "suitable" matching.
  + Participants rate members of opposite sex.
  + Each man lists women in order of preference from best to worst.
  + Each woman lists men inorder of preference from best to worst.
+ Perfect matching: everyone is matched monogamously
  + Each man gets exactly one woman.
  + Each woman gets exactly one man.
+ Stability: no incentive for some pair of participants to undermine assignment by joint action.
  + In matching M, an unmatched pair m-w is unstable if man m and woman w prefer each other to current partners.
  + Unstable pair m-w could each improve by eloping.
+ Stable matching: prefer matching with no unstable pairs.
+ Stable matching problem. Given the preference lists of n men and n women, find a stable matching if one exists.

这个问题不一定有稳定匹配的结果。

## 3. Propose And Reject Algorithm

又称为GS算法，由Gale Shapley提出。是一个直观的能保证找到稳定排序的方法

```
Initialize each person to be free.
while(some man is free and hasn't proposed to every woman){
    Choose such a man m
    w = first woman on m's list to whom m has not yet proposed 
    if (w is free)
    	assign m and w to be engaged
    else if (w prefers m to her fiance m')
    	assign m and w to be engaged, and m' to be free
    else
    	w rejects m
}
```

简单来说，就是每次循环随机选择一个男生，这个男生根据自己的喜欢程度，向最喜欢的女生表白，如果那个女生单身，就表白成功；如果那个女生有男朋友，但是更喜欢表白的这个男生，就成功挖墙脚；如果那个女生有男朋友，且更喜欢现在的男朋友，这个男生就被发好人卡了，并且再也不爱这个女生了（指将这个女生从列表中去掉）。

我们观察到两点：男生依照喜欢的程度表白，而女生只要脱单了就不会单身了。

由于有n个男生，每个男生最多表白n次（n个女生），所以算法的时间复杂度为O($n^2$ )。



证明：所有的男生和女生都匹配了

​	利用反证法，如果有一个男生和女生还在演孤独传说，女生由于单身，所以一定没有被表白过，但是那个男生向所有人表白了，因为他没有匹配到，所以不存在未匹配的情况。



证明：没有不稳定的cp

​	利用反证法，如果男生Stefan和女生Elena是不稳定的一对，他们互相喜欢胜过GS排序中得到的伴侣。此时有两种情况：Stefan向Elena表白过，则Elena拒绝了Stefan（当场或者未来），Elena更喜欢他的现伴侣；Stefan没有向Elena表白过，则Stefan更喜欢现在的伴侣。两种情况都说明他们应该是稳定的普通朋友。

## 4. Efficient Implementation

当一个男生向一个女生表白且这个女生有男朋友的时候，我们需要快速知道她更爱谁，所以我们可以存储下女生对每个男生的喜欢排名。

```
for i = 1 to n
	inverse[pref[i]] = i
```

## 5. Understanding the Solution

对于一个给定的问题，可能会存在多种稳定匹配，是否所有通过gs算法得到的匹配结果都是意义的呢？

定义：如果男生Olega和女生Susan在多个稳定匹配中都是伴侣，那么称Olega是Susan的有效伴侣。

男生最佳分配：方案中男生都找到了自己的最佳有效伴侣

证明：gs算法对于每个男生来说都是最佳分配

​	假设一个男生没有得到最佳的分配对象，由于他是按照喜欢程度表的白，如果他现在的女朋友排喜欢程度的第i位，则i位以前的绝对追不到，如果侥幸追到了，那个女生也必然会有更好的男友选择。

男生追到的女生就是最好选择，而女生是选择了稳定匹配中能得到的最差选择。

所以我们得出结论，主动表白才是得到最好选择的方法。

