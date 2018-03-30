#!/usr/bin/python -w
#--encoding: gbk -*---

import sys
import math
import re
import string

ERR_DIFF = 0.1
MAX = 32
def err(ranklist, size):
	if len(ranklist) < size:
		return -1
	s = 0.0
	p = 1.0
	for i in range(size):
		r = (math.pow(2.0, ranklist[i]) - 1)*1.0/MAX
		s += p*r/(i+1)
		p *= (1.0 - r)
	return s
	
def dcg(ranklist, size):
	if len(ranklist) < size:
		return -1
	s = 0.0
	for i in range(size):
		s += (math.pow(2.0, ranklist[i]) - 1)*1.0/math.log(2+i,2)
	return s

def kendall(ranklist, featureValue):
	calcLen = min(len(ranklist), len(featureValue))
	concordant = 0.0
	discordant = 0.0
	for i in range(0, calcLen):
		label = ranklist[i]
		fv = featureValue[i]
		for j in range(i + 1, calcLen):
			if math.fabs(fv - featureValue[j]) < 0.0001:
				continue
			if label < ranklist[j]:
				discordant += 1
			if label > ranklist[j]:
				concordant += 1

	tmp = calcLen * (calcLen - 1) or 1

	return (concordant - discordant) * 2 / tmp

def corr(fealist, scorelist, size):
	fea_sum = 0.0
	fea_square = 0.0
	score_sum = 0.0
	score_square = 0.0
	fs_mul = 0.0
	for i in range(size):
		feature = float(fealist[i])
		score = float(scorelist[i])
		fea_sum += feature
		fea_square += feature * feature
		score_sum += score
		score_square += score * score
		fs_mul += feature * score
	up = size*fs_mul - fea_sum*score_sum
	down = (size*fea_square - fea_sum*fea_sum) * (size*score_square - score_sum*score_sum)
	if down > 0:
		return up/math.sqrt(down)
	return 0

def mycmp(x,y):
	return float(x) > float(y)
	
def compare(cmpMode):
	labelfile = ""
	featurenamefile = ""
	targetfid = ""
	datafileTest = ""
	datafileBase = ""
	doclimit = 0
	summaryoutput = ""
	detailoutput = ""
	if cmpMode == 0:
		labelfile = sys.argv[1]
		featurenamefile = sys.argv[3]
		datafileTest = sys.argv[4] #前2000条结果的特征文件
		datafileBase = sys.argv[5] #前2000条结果的特征文件
		doclimit = int(sys.argv[6])
		summaryoutput = sys.argv[7]
		detailoutput = sys.argv[8]
		unlabeloutput = sys.argv[9]
		fealookupfile = sys.argv[10]
	elif cmpMode == 1:
		labelfile = sys.argv[1]
		targetfid = sys.argv[3]
		featurenamefile = sys.argv[4]
		datafileTest = sys.argv[5] #前2000条结果的特征文件
		doclimit = int(sys.argv[6])
		summaryoutput = sys.argv[7]
		detailoutput = sys.argv[8]
		unlabeloutput = sys.argv[9]
		fealookupfile = sys.argv[10]
	
	"""读入要分析的featureid-featurename"""
	feature_name = {}
	for line in open(featurenamefile).readlines():
		items = line.strip().split("\t")
		feature_name[items[0]] = items[1]
		
	"""读取标注文件"""
	query_reslabel = {}
	for line in open(labelfile).readlines():
		items = line.strip().split("\t")
		try:
			if len(items) == 3:
				query_reslabel[items[0]+"\t"+items[1]] = int(items[2])
		except:
			pass
	print "read featurename and labelfile ok"
	
	dic_lookup = {}
	#for line in open(fealookupfile).readlines():
	#	items = line.strip().split("\t")
	#	dic_lookup[items[0]] = items[1]
	#	print "look"+items[0]

	out_detail = open(detailoutput, "w")
	out_summary = open(summaryoutput, "w")
	#out_unlabel = open(unlabeloutput,"w")
	
	"""读入test文件，按特征排序"""
	print "processing test"+datafileTest	
	print "doclimit="+str(doclimit)
	
	query_feature_metrics_test = {} #query_feature_metric[query][feaid] = [err1,err3,dcg1,dcg3,goodpos,goodnum,labelnum]
	query_feature_poslist_test = {} #query_feature_poslist[query][feaid] = [g1_pos, g2_pos...]
	query_fvalue_test = {} #query_fvalue[query][feaid] = [f1,f2...]
	query_dscore_test = {} #query_dscore[query][feaid] = [s1,s2...]
	feature_validquery ={} 
	query = ""
	qid = ""
	rescount = 0
	linenum = 0
	featureSequence = {}
	for line in open(datafileTest).readlines():
		linenum += 1
		if linenum%100000 == 0:
			print linenum
		items = line.strip().split("\t")
		if len(items) < 2:
			continue
		detail = items[0].split(" ")
		resDocID = items[1]
		resDocPos = items[2]
		resScore = detail[0]
		curquery = ""
		curqid = ""
		m = re.search('#(.*)$', detail[len(detail)-1])
		if m != None:
			curquery = m.group(0).split("#")[1]
			curqid = detail[1].split(":")[1]
		if curquery == "":
			continue
		query_reslabel[curquery +"\t" + resDocID] = float(resScore)
		
		"""处理每个query"""
		if curquery != query or curqid != qid:
			if query != "":
				query_dscore_test[query] = {}
				query_fvalue_test[query] = {}
				query_feature_metrics_test[query] = {}
				query_feature_poslist_test[query] = {}
				for feaid in featureSequence.keys():
					"""按照特征取值重排序"""
					docsorted=sorted(featureSequence[feaid].items(), key=lambda value:float(value[0]), reverse=True)
					if feaid == '1337': #for miss_cut_rate
						docsorted=sorted(featureSequence[feaid].items(),key=lambda value:float(value[0]),reverse=False)
					"""计算有标注结果的指标"""
					newrankpos = -1
					ranklist_dscore = [] #标注的列表
					goodpos = -1
					goodposlist = []
					labeledNum = 0
					goodNum = 0
					featurenozeroNum = 0
					firstNozeroPos = -1
					firstGoodPos = -1
					for docfeature in docsorted:
						tmp_docfeature = {}
						for k in range(len(docfeature[1])):
							key = query + "\t" + docfeature[1][k]
							if key in query_reslabel:
								tmp_docfeature[key] = query_reslabel[key]
							else:
								tmp_docfeature[key] = -1
						#按照最坏label情况排序
						tmp_docsorted = sorted(tmp_docfeature.items(), key=lambda value:float(value[1]), reverse=False)
						for qd in tmp_docsorted:
							newrankpos += 1
							#if feaid in dic_lookup:
								#out_unlabel.write(str(feaid)+"\t"+str(qd[0])+"\t"+str(qd[1])+"\t"+str(docfeature[0])+"\t"+str(newrankpos)+"\ttest\n")
							if qd[1] != -1:
								labeledNum += 1
								ranklist_dscore.append(qd[1]) #标注的列表
								query_dscore_test[query].setdefault(feaid, []).append(qd[1]) #标注的列表
								query_fvalue_test[query].setdefault(feaid, []).append(docfeature[0]) #特征值的列表
								out_detail.write(str(feaid)+"\t"+str(qd[0])+"\t"+str(qd[1])+"\t"+str(docfeature[0])+"\t"+str(newrankpos)+"\ttest\n")
								if qd[1] >= 2.99: #good doc
									goodNum += 1
									goodpos += newrankpos
									goodposlist.append(newrankpos)
									if firstNozeroPos == -1:
										firstNozeroPos = newrankpos
									if firstGoodPos == -1 and qd[1] >= 3.99:
										firstGoodPos = newrankpos
								if float(docfeature[0]) > 0: #非零的feature
									featurenozeroNum += 1
					if goodNum == 0: 
						goodpos = doclimit
						firstNozeroPos = 1.0/doclimit
						firstGoodPos = 1.0/doclimit
					else:
						goodpos += 1
						firstNozeroPos = 1.0/(1+firstNozeroPos)
						if firstGoodPos == -1:
							firstGoodPos = 1.0/doclimit
						else:
							firstGoodPos = 1.0/(1+firstGoodPos)
					metrics = {'err1':0, 'err3':0, 'err5':0, 'dcg5':0, 'goodpos':goodpos, 'goodNum':goodNum, 'labeledNum':labeledNum, 'fnp':firstNozeroPos, 'fgp':firstGoodPos, 'kendall':0}
					#metrics = [0,0,0,0,goodpos,goodNum,labeledNum,firstNozeroPos,firstGoodPos]
					
					# """只考虑标注结果多于3条，且特征不全为0的查询"""
					# if labeledNum >= 3 and featurenozeroNum > 0:
						# metrics[0] = err(ranklist_dscore, 1)
						# metrics[1] = err(ranklist_dscore, 3)
						# metrics[2] = dcg(ranklist_dscore, 1)
						# metrics[3] = dcg(ranklist_dscore, 3)
						# if feaid not in feature_validquery:
							# feature_validquery[feaid] = {}
						# feature_validquery[feaid][query] = 1
					# else:
						# ranklist_dscore.extend([0,0,0]); #补充不足三条标注的结果
						# metrics[0] = err(ranklist_dscore, 1)
						# metrics[1] = err(ranklist_dscore, 3)
						# metrics[2] = dcg(ranklist_dscore, 1)
						# metrics[3] = dcg(ranklist_dscore, 3)
					# query_feature_metrics_test[query][feaid] = metrics
					# query_feature_poslist_test[query][feaid] = goodposlist
					
					"""所有查询都考虑"""
					if feaid not in feature_validquery:
						feature_validquery[feaid] = {}
					feature_validquery[feaid][query] = 1
					#ranklist_dscore.extend([0,0,0,0,0]); #补充不足三条标注的结果
					metrics['err1'] = err(ranklist_dscore, 1)
					metrics['err3'] = err(ranklist_dscore, 3)
					metrics['err5'] = err(ranklist_dscore, 5)
					metrics['dcg5'] = dcg(ranklist_dscore, 5)
					metrics['kendall'] = kendall(ranklist_dscore,  [float(v) for v in query_fvalue_test[query][feaid]])
					query_feature_metrics_test[query][feaid] = metrics
					query_feature_poslist_test[query][feaid] = goodposlist
					
			query = curquery
			qid = curqid
			rescount = 0
			featureSequence = {}
				
		"""处理每条结果"""
		if rescount >= doclimit:
			continue
		rescount = int(resDocPos)+1
		for i in range(2, len(detail)-1):
			try:
				feaid = detail[i].split(":")[0]
				feavalue = detail[i].split(":")[1]
				if feaid not in feature_name:
					continue
				if feaid not in featureSequence:
					featureSequence[feaid] = {}
				featureSequence[feaid].setdefault(feavalue, []).append(resDocID)
			except:
                                #print detail[i]
				pass
	
	#####################################################################################################
	if cmpMode == 0:
		"""读入base文件"""
		query_fvalue_base = {} #query_fvalue[query][feaid] = [f1,f2...]
		query_dscore_base = {} #query_dscore[query][feaid] = [s1,s2...]
		query_feature_metrics_base = {}
		query_feature_poslist_base = {}
		featureSequence = {}
		rescount = 0
		query = ""
		qid = ""
		linenum = 0
		print "processing base"+datafileBase
		for line in open(datafileBase).readlines():
			linenum += 1
			if linenum%100000 == 0:
				print linenum
			items = line.strip().split("\t")
			if len(items) < 2:
				continue
			detail = items[0].split(" ")
			resDocID = items[1].replace(",", "")
			resDocPos = items[2].replace(",", "")
			resScore = detail[0]
			curquery = ""
			curqid = ""
			m = re.search('#(.*)$', detail[len(detail)-1])
			if m != None:
				curquery = m.group(0).split("#")[1]
				curqid = detail[1].split(":")[1]
			if curquery == "":
				continue

			"""处理每个query"""
			if curquery != query or curqid != qid:
				if query != "":
					query_dscore_base[query] = {}
					query_fvalue_base[query] = {}
					query_feature_metrics_base[query] = {}
					query_feature_poslist_base[query] = {}
					for feaid in featureSequence.keys():
						"""按照特征取值重排序结果"""
						docsorted=sorted(featureSequence[feaid].items(), key=lambda value:float(value[0]), reverse=True)
						if feaid == '1337':
							docsorted=sorted(featureSequence[feaid].items(),key=lambda value:float(value[0]),reverse=False)
						"""计算有标注结果的指标"""
						newrankpos = -1
						ranklist_dscore = []
						goodpos = -1
						goodposlist = []
						labeledNum = 0
						goodNum = 0
						featurenozeroNum = 0
						firstNozeroPos = -1
						firstGoodPos = -1
						for docfeature in docsorted:
							tmp_docfeature = {}
							for k in range(len(docfeature[1])):
								key = query + "\t" + docfeature[1][k]
								if key in query_reslabel:
									tmp_docfeature[key] = query_reslabel[key]
								else:
									tmp_docfeature[key] = -1
							#按照最坏情况排序
							tmp_docsorted = sorted(tmp_docfeature.items(), key=lambda value:float(value[1]), reverse=False)
							for qd in tmp_docsorted:
								newrankpos += 1
#								if feaid in dic_lookup:
#									out_unlabel.write(str(feaid)+"\t"+str(qd[0])+"\t"+str(qd[1])+"\t"+str(docfeature[0])+"\t"+str(newrankpos)+"\tbase\n")
								if qd[1] != -1:
									labeledNum += 1
									ranklist_dscore.append(qd[1]) #标注的列表
									query_dscore_base[query].setdefault(feaid, []).append(qd[1]) #标注的列表
									query_fvalue_base[query].setdefault(feaid, []).append(docfeature[0]) #特征值的列表									
									out_detail.write(str(feaid)+"\t"+str(qd[0])+"\t"+str(qd[1])+"\t"+str(docfeature[0])+"\t"+str(newrankpos)+"\tbase\n")
									if qd[1] > 2.99:
										goodNum += 1
										goodpos += newrankpos
										goodposlist.append(newrankpos)
										if firstNozeroPos == -1: #首条非0结果位置
											firstNozeroPos = newrankpos
										if firstGoodPos == -1 and qd[1] >= 3.99: #首条3分结果位置
											firstGoodPos = newrankpos
									if float(docfeature[0]) > 0:
										featurenozeroNum += 1

						if goodNum == 0: 
							goodpos = doclimit
							firstNozeroPos = 1.0/doclimit
							firstGoodPos = 1.0/doclimit
						else:
							goodpos += 1
							firstNozeroPos = 1.0/(1+firstNozeroPos)
							if firstGoodPos == -1:
								firstGoodPos = 1.0/doclimit
							else:
								firstGoodPos = 1.0/(1+firstGoodPos)
						metrics = {'err1':0, 'err3':0, 'err5':0, 'dcg5':0, 'goodpos':goodpos, 'goodNum':goodNum, 'labeledNum':labeledNum, 'fnp':firstNozeroPos, 'fgp':firstGoodPos, 'kendall':0}
						#metrics = [0,0,0,0,goodpos,goodNum,labeledNum,firstNozeroPos,firstGoodPos]
						# """结果多于3条，且特征不全为0的查询"""
						# if labeledNum >= 3 and featurenozeroNum > 0:
							# metrics[0] = err(ranklist_dscore, 1)
							# metrics[1] = err(ranklist_dscore, 3)
							# metrics[2] = dcg(ranklist_dscore, 1)
							# metrics[3] = dcg(ranklist_dscore, 3)
							# if feaid not in feature_validquery:
								# feature_validquery[feaid] = {}
							# feature_validquery[feaid][query] = 1
						# else:
							# ranklist_dscore.extend([0,0,0])
							# metrics[0] = err(ranklist_dscore, 1)
							# metrics[1] = err(ranklist_dscore, 3)
							# metrics[2] = dcg(ranklist_dscore, 1)
							# metrics[3] = dcg(ranklist_dscore, 3)
						# query_feature_metrics_base[query][feaid] = metrics
						# query_feature_poslist_base[query][feaid] = goodlist
						if feaid not in feature_validquery:
							feature_validquery[feaid] = {}
						feature_validquery[feaid][query] = 1
						#ranklist_dscore.extend([0,0,0,0,0])
						metrics['err1'] = err(ranklist_dscore, 1)
						metrics['err3'] = err(ranklist_dscore, 3)
						metrics['err5'] = err(ranklist_dscore, 5)
						metrics['dcg5'] = dcg(ranklist_dscore, 5)
						try:
							metrics['kendall'] = kendall(ranklist_dscore,  [float(v) for v in query_fvalue_base[query][feaid]])
						except:
							metrics['kendall'] = 0
						query_feature_metrics_base[query][feaid] = metrics
						query_feature_poslist_base[query][feaid] = goodposlist						
				query = curquery
				qid = curqid
				rescount = 0
				featureSequence = {}
			
			"""处理每条结果"""
			if rescount >= doclimit:
				continue
			#rescount += 1
			rescount = int(resDocPos)+1
			for i in range(2, len(detail)-1):
				try:
					feaid = detail[i].split(":")[0]
					feavalue = detail[i].split(":")[1]
					if feaid not in feature_name:
						continue
					if feaid not in featureSequence:
						featureSequence[feaid] = {}
					featureSequence[feaid].setdefault(feavalue, []).append(resDocID)
				except:
					#print detail[i]
					pass

		"""取validquery进行对比"""
		print "compare test and base"
		metricsName = ["err1","err3","err5","dcg5","goodpos","goodNum","labeledNum","fnp","fgp", "kendall"]
		for feaid in feature_name.keys():
			feastring = feature_name[feaid]
			querynum = 0
			test_docscorelist = []
			test_feavaluelist = []
			base_docscorelist = []
			base_feavaluelist = []
			test_all = [0] * len(metricsName) #用于统计各个metrics的值
			base_all = [0] * len(metricsName)
			qnum_all = [0] * len(metricsName)
			win = [0] * len(metricsName)
			lose = [0] * len(metricsName)
			draw = [0] * len(metricsName)
			winratio = [0] * len(metricsName)
			if feaid not in feature_validquery:
				continue
			for q in feature_validquery[feaid].keys():
				if q not in query_dscore_base or feaid not in query_dscore_base[q]:
					continue
				if q not in query_dscore_test or feaid not in query_dscore_test[q]:
					continue
				querynum += 1
				test_metrics = query_feature_metrics_test[q][feaid]
				base_metrics = query_feature_metrics_base[q][feaid]
				"""计算好结果平均位置"""
				minGoodNum = test_metrics['goodNum']
				if base_metrics['goodNum'] < minGoodNum:
					minGoodNum = base_metrics['goodNum']
				if minGoodNum > 0:
					test_posadd = 0
					base_posadd = 0
					for i in range(minGoodNum):
						test_posadd += query_feature_poslist_test[q][feaid][i]
						base_posadd += query_feature_poslist_base[q][feaid][i]
					test_metrics['goodpos'] = 1.0*(test_posadd / minGoodNum)
					base_metrics['goodpos'] = 1.0*(base_posadd/ minGoodNum)
				out_detail.write(feastring+"\t"+q+"\t"+"test\t"+str(test_metrics)+"\n")
				out_detail.write(feastring+"\t"+q+"\t"+"base\t"+str(base_metrics)+"\n")
				"""计算统计特征"""
				for i in range(9):
					if test_metrics[metricsName[i]] >= 0 and base_metrics[metricsName[i]] >= 0: 
						test_all[i] += test_metrics[metricsName[i]]	
						base_all[i] += base_metrics[metricsName[i]]
						qnum_all[i] += 1
				test_all[9] += test_metrics[metricsName[9]]
				base_all[9] += base_metrics[metricsName[9]]
				qnum_all[9] += 1
				#得到全局的特征和得分序列，后面计算correlation
				if feaid in query_dscore_base[q]:
					base_docscorelist.extend(query_dscore_base[q][feaid])
					base_feavaluelist.extend(query_fvalue_base[q][feaid])
				if feaid in query_dscore_test[q]:
					test_docscorelist.extend(query_dscore_test[q][feaid])
					test_feavaluelist.extend(query_fvalue_test[q][feaid])
							
				"""计算胜出落败"""
				for i in range(4):
					if test_metrics[metricsName[i]] >= 0 and base_metrics[metricsName[i]] >= 0:
						if test_metrics[metricsName[i]] - base_metrics[metricsName[i]] > 0.05:
							win[i] += 1
						elif base_metrics[metricsName[i]] - test_metrics[metricsName[i]] > 0.05:
							lose[i] += 1
						else:
							draw[i] += 1
				#for avgpos
				if test_metrics['goodpos'] - base_metrics['goodpos'] > 5:
					lose[4] += 1
				elif base_metrics['goodpos'] - test_metrics['goodpos'] > 5:
					win[4] += 1
				else:
					draw[4] += 1
				#for gooddocNum
				if test_metrics['goodNum'] - base_metrics['goodNum'] > 2:
					win[5] += 1
				elif base_metrics['goodNum'] - test_metrics['goodNum'] > 2:
					lose[5] += 1
				else:
					draw[5] += 1
			
			correlation_base = corr(base_docscorelist, base_feavaluelist, base_all[6])
			correlation_test = corr(test_docscorelist, test_feavaluelist, test_all[6])
			
			if querynum == 0:
				out_summary.write(feastring+"\tnoquery\n")
				continue
			for i in range(len(metricsName)):
				if qnum_all[i] > 0:
					test_all[i] = 1.0*test_all[i]/qnum_all[i]
					base_all[i] = 1.0*base_all[i]/qnum_all[i]
			for i in range(6):
				if win[i] != lose[i]:
					winratio[i] = 1.0*(win[i] + draw[i]/2)/(win[i]+draw[i]+lose[i])
				else:
					winratio[i] = 0.5
							
			out_summary.write(feastring+"\t"+str(querynum)+"\n")
			for i in range(6):
				out_summary.write("\t"+metricsName[i]+"\t"+str(qnum_all[i])+"\t"+str(test_all[i])+"\t"+str(base_all[i])+"\t"+str(win[i])+"\t"+str(draw[i])+"\t"+str(lose[i])+"\t"+str(winratio[i])+"\n")
			out_summary.write("\tlabeledNum"+"\t"+str(qnum_all[6])+"\t"+str(test_all[6])+"\t"+str(base_all[6])+"\n")
			out_summary.write("\t[MRR2]"+metricsName[7]+"\t"+str(qnum_all[7])+"\t"+str(test_all[7])+"\t"+str(base_all[7])+"\n")
			out_summary.write("\t[MRR3]"+metricsName[8]+"\t"+str(qnum_all[8])+"\t"+str(test_all[8])+"\t"+str(base_all[8])+"\n")
			out_summary.write("\t"+metricsName[9]+"\t"+str(qnum_all[9])+"\t"+str(test_all[9])+"\t"+str(base_all[9])+"\n")
			out_summary.write("\tcorrelation\t"+ str(correlation_test)+"\t"+str(correlation_base)+"\n")
			
	############################################################################################
	elif cmpMode == 1:
		print "compare feature "+str(targetfid)
		metricsName = ["err1","err3","err5","dcg5","goodpos","goodNum","labeledNum","fnp","fgp", "kendall"]
		targetfeastring = feature_name[targetfid]
		#和每个特征比较
		for feaid in feature_name.keys():
			if feaid == targetfid:
				continue
			feastring = feature_name[feaid]
			querynum = 0
			base_docscorelist = []
			test_docscorelist = []
			base_feavaluelist = []
			test_feavaluelist = []
			test_all = [0] * len(metricsName)
			base_all = [0] * len(metricsName)
			qnum_all = [0] * len(metricsName)
			win = [0] * len(metricsName)
			lose = [0] * len(metricsName)
			draw = [0] * len(metricsName)
			winratio = [0] * len(metricsName)
			if feaid not in feature_validquery:
				continue
			for q in feature_validquery[feaid].keys():
				if q not in query_dscore_test or feaid not in query_dscore_test[q]:
					continue
				querynum += 1
				base_metrics = query_feature_metrics_test[q][feaid]
				test_metrics = query_feature_metrics_test[q][targetfid]
				"""计算好结果平均位置"""
				ori_goodpos = test_metrics['goodpos']
				minGoodNum = test_metrics['goodNum']
				if base_metrics['goodNum'] < minGoodNum:
					minGoodNum = base_metrics['goodNum']
				if minGoodNum > 0:
					test_posadd = 0
					base_posadd = 0
					for i in range(minGoodNum):
						test_posadd += query_feature_poslist_test[q][targetfid][i]
						base_posadd += query_feature_poslist_test[q][feaid][i]
					test_metrics['goodpos'] = 1.0*(test_posadd / minGoodNum)
					base_metrics['goodpos'] = 1.0*(base_posadd/ minGoodNum)
				
				out_detail.write(targetfeastring+"\t"+q+"\t"+"test\t"+str(test_metrics)+"\n")
				out_detail.write(feastring+"\t"+q+"\t"+"base\t"+str(base_metrics)+"\n")
				
				"""计算统计特征"""
				for i in range(9):
					if test_metrics[metricsName[i]] >=0 and base_metrics[metricsName[i]] >=0: 
						test_all[i] += test_metrics[metricsName[i]]
						base_all[i] += base_metrics[metricsName[i]]
						qnum_all[i] += 1
				test_all[9] += test_metrics[metricsName[9]]
				base_all[9] += base_metrics[metricsName[9]]
				qnum_all[9] += 1
				#得到全局的特征和得分序列，后面计算correlation
				if feaid in query_dscore_test[q]:
					base_docscorelist.extend(query_dscore_test[q][feaid])
					base_feavaluelist.extend(query_fvalue_test[q][feaid])
				if targetfid in query_dscore_test[q]:
					test_docscorelist.extend(query_dscore_test[q][targetfid])
					test_feavaluelist.extend(query_fvalue_test[q][targetfid])
							
				"""计算胜出落败"""
				for i in range(4):
					if test_metrics[metricsName[i]] >=0 and base_metrics[metricsName[i]] >=0:
						if test_metrics[metricsName[i]] - base_metrics[metricsName[i]] > 0.05:
							win[i] += 1
						elif base_metrics[metricsName[i]] - test_metrics[metricsName[i]] > 0.05:
							lose[i] += 1
						else:
							draw[i] += 1
				#for avgpos
				if test_metrics['goodpos'] - base_metrics['goodpos'] > 5:
					lose[4] += 1
				elif base_metrics['goodpos'] - test_metrics['goodpos'] > 5:
					win[4] += 1
				else:
					draw[4] += 1
				#for gooddocNum
				if test_metrics['goodNum'] - base_metrics['goodNum'] > 2:
					win[5] += 1
				elif base_metrics['goodNum'] - test_metrics['goodNum'] > 2:
					lose[5] += 1
				else:
					draw[5] += 1
				
				#恢复goodpos的值
				test_metrics['goodpos'] = ori_goodpos
			
			correlation_base = corr(base_docscorelist, base_feavaluelist, base_all[6])
			correlation_test = corr(test_docscorelist, test_feavaluelist, test_all[6])
			
			if querynum == 0:
				out_summary.write(feastring+"\tnoquery\n")
				continue
			for i in range(len(metricsName)):
				if qnum_all[i] > 0:
					test_all[i] = 1.0*test_all[i]/qnum_all[i]
					base_all[i] = 1.0*base_all[i]/qnum_all[i]
			for i in range(6):
				if win[i] != lose[i]:
					winratio[i] = 1.0*(win[i] + draw[i]/2)/(win[i]+draw[i]+lose[i])
				else:
					winratio[i] = 0.5
			
			out_summary.write(feastring+"\t"+str(querynum)+"\n")
			for i in range(6):
				out_summary.write("\t"+metricsName[i]+"\t"+str(qnum_all[i])+"\t"+str(test_all[i])+"\t"+str(base_all[i])+"\t"+str(win[i])+"\t"+str(draw[i])+"\t"+str(lose[i])+"\t"+str(winratio[i])+"\n")
			out_summary.write("\tlabeledNum"+"\t"+str(qnum_all[6])+"\t"+str(test_all[6])+"\t"+str(base_all[6])+"\n")
			out_summary.write("\t[MRR2]"+metricsName[7]+"\t"+str(qnum_all[7])+"\t"+str(test_all[7])+"\t"+str(base_all[7])+"\n")
			out_summary.write("\t[MRR3]"+metricsName[8]+"\t"+str(qnum_all[8])+"\t"+str(test_all[8])+"\t"+str(base_all[8])+"\n")
			out_summary.write("\t"+metricsName[9]+"\t"+str(qnum_all[9])+"\t"+str(test_all[9])+"\t"+str(base_all[9])+"\n")
			out_summary.write("\tcorrelation\t"+ str(correlation_test)+"\t"+str(correlation_base)+"\n")
			
			
def usage():
	print """Usage:\n./RankbyFeatureComp.py labelfile[in] -c featureName[in] testdata[in] basedata[in] docnumlimit[in] result[out] detailinfo[out]\n"""
	print """./RankbyFeatureComp.py labelfile[in] -f [fid] featureName[in] testdata[in] docnumlimit[in] result[out] detailinfo[out]\n"""
	print """Format[result]:\nfeatureName querynum"""
	print """\terr1 test_avg base_avg win draw lose winratio"""
	print """\terr3 test_avg base_avg win draw lose winratio"""
	print """\tdcg1 test_avg base_avg win draw lose winratio"""
	print """\tdcg3 test_avg base_avg win draw lose winratio"""
	print """\tgoodpos test_avg base_avg win draw lose winratio"""
	print """\tgoodNum test_avg base_avg win draw lose winratio"""
	print """\tlabeledNum test_avg base_avg"""
	print """\t[MRR2]fnp test_avg base_avg"""
	print """\t[MRR3]fgp test_avg base_avg"""
	print """\tcorrelation test_avg base_avg"""
	
if __name__ == "__main__":
	if len(sys.argv) <= 7:
		usage()
		exit(0)
	print sys.argv[0]+" "+sys.argv[1] + " "+sys.argv[2] + " "+sys.argv[3]
	print sys.argv[4]
	print sys.argv[5]
	print sys.argv[6]
	print sys.argv[7]
	print sys.argv[8]
	if sys.argv[2] == '-c':
		compare(0)
	if sys.argv[2] == '-f':
		compare(1)
