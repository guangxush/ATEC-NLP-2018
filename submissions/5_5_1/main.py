#-*- coding: utf-8 -*-
#coding:utf-8
#import jieba
import sys
reload(sys)
sys.setdefaultencoding("gbk")
import math


d = {}
log = lambda x: float('-inf') if not x else math.log(x)
prob = lambda x: d[x] if x in d else 0 if len(x)>1 else 1
 
def init(filename='SogouLabDic.dic'):
	d['_t_'] = 0.0
	with open(filename, 'r') as handle:
		for line in handle:
			word, freq = line.split('\t')[0:2]
			d['_t_'] += int(freq)+1
			try:
				d[word.decode('gbk')] = int(freq)+1
			except:
				d[word] = int(freq)+1
 
def solve(s):
	l = len(s)
	p = [0 for i in range(l+1)]
	t = [0 for i in range(l)]
	for i in xrange(l-1, -1, -1):
		p[i], t[i] = max((log(prob(s[i:i+k])/d['_t_'])+p[i+k], k)
						for k in xrange(1, l-i+1))
	while p[l]<l:
		yield s[p[l]:p[l]+t[p[l]]]
		p[l] += t[p[l]]

def cos_dist(a, b):
	if len(a) != len(b):
		return None
	part_up = 0.0
	a_sq = 0.0
	b_sq = 0.0
	for a1, b1 in zip(a,b):
		part_up += a1*b1
		a_sq += a1**2
		b_sq += b1**2
	part_down = math.sqrt(a_sq*b_sq)
	if part_down == 0.0:
		return None
	else:
		return part_up / part_down


def process(inpath, outpath):
	init()
	with open(inpath, 'r') as fin, open(outpath, 'w') as fout:
		for line in fin:
			lineno, sen1, sen2 = line.strip().split('\t')
			print sen1,sen2
			s1=list(solve(sen1))
			s2=list(solve(sen2))
			key=list(set(s1+s2))
			keyLen=len(key)
			keyValue=0
			sk1=[keyValue]*keyLen
			sk2=[keyValue]*keyLen
			for index,keyElement in enumerate(key):
				if keyElement in s1:
					sk1[index]=sk1[index]+1
				if keyElement in s2:
					sk2[index]=sk2[index]+1 
			r=cos_dist(sk1,sk2) 
			if r > 0.8:
				fout.write(lineno + '\t1\n')
			else:
				fout.write(lineno + '\t0\n')

if __name__ == '__main__':
	process(sys.argv[1], sys.argv[2])
