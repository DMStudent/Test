#!/usr/bin/python

import re
import sys

if __name__ == '__main__':
  assert len(sys.argv) == 4
  #data_path = sys.argv[1]
  #input_file = sys.argv[2]
  #merge_file = data_path + '/merge_data'
  #new_file = data_path + '/new_data'
  fake_output = sys.argv[1]
  merge_file = sys.argv[2]
  new_file = sys.argv[3]

  qt_to_score = {}
  ifs = open(fake_output,'r')
  line_num = 0
  while True:
    line = ifs.readline()
    if line == '':
      break
    line_num += 1
    line = line.rstrip()
    tokens = line.split('\t')
    if len(tokens) == 3:
      hidden_query = tokens[0]
      hidden_title = tokens[1]
      score = float(tokens[2])
      hidden_qt = hidden_query + '\t' + hidden_title
      if not qt_to_score.has_key(hidden_qt):
        qt_to_score[hidden_qt] = score
      else:
        if ((score - qt_to_score[hidden_qt]) > 0.0001 or (score - qt_to_score[hidden_qt]) < -0.0001):
          sys.stderr.write('Error: qt-"%s" has different scores at line %d, %.6f-%.6f\n'%(hidden_qt,line_num,qt_to_score[hidden_qt],  score))
    #else:
    #  sys.stderr.write('ERROR: token != 3 at line %d.\n'%line_num)
  ifs.close()


  hidden_to_qd = {}
  qd_to_hidden = {}
  qd_to_qdc = {}
  ifs = open(new_file,'r')
  line_num = 0
  while True:
    line = ifs.readline()
    if line == '':
      break
    line_num += 1
    line = line.rstrip()
    tokens = line.split('\t')

    assert len(tokens) == 5
    query_str = re.sub('^originalQueryString:','',tokens[0])
    doc_id = re.sub('^docid:','',tokens[1])
    hidden_query = re.sub('^query_termid:','',tokens[2])
    hidden_title = re.sub('^cnn_title:','',tokens[4])

    if hidden_query != '' and hidden_title != '' and query_str != '' and doc_id != '':
      hidden_qt = hidden_query + '\t' + hidden_title
      qd = query_str + '\t' + doc_id
      qdc = tokens[2] + '\t' + tokens[3] + '\t' + tokens[4]
      qd_to_hidden[qd] = hidden_qt
      qd_to_qdc[qd] = qdc
      #sys.stdout.write('originalQueryString:%s\tdocid:%s\t%s\t6004:%.6f\n'%(query_str,doc_id,qd_to_qdc[qd],qt_to_score[hidden_qt]))

  ifs.close()


  ifs = open(merge_file, 'r')
  line_num = 0
  fail_count = 0
  total_count = 0
  while True:
    line = ifs.readline()
    if line == '':
      break
    line_num += 1
    line = line.rstrip()
    line = re.sub(',','',line)
    tokens = line.split('\t')
    assert len(tokens) == 3
    doc_id = tokens[1]
    feature_line = tokens[0]
    #feature_line = tokens[0]
    query_str = re.sub('^.+ #(.+)$',r'\1',feature_line)
    fea_5004 = float(re.sub('^.+ 5004:(.+?) .+$',r'\1',feature_line))
    #if fea_5004 == 0 or fea_5004 > 1.0:
    #  continue
    qd = query_str + '\t' + doc_id
    if qd_to_hidden.has_key(qd):
      hidden_qt = qd_to_hidden[qd]
      if not qt_to_score.has_key(hidden_qt):
        sys.stderr.write('Error at line %d: hidden_qt-"%s" has no score\n'%(line_num,hidden_qt))
        continue
      score_test = qt_to_score[hidden_qt]
      sys.stdout.write(re.sub(' 5004:.+? ',' 5004:%.6f '%score_test,line) + '\n')
      if score_test - fea_5004 > 0.01 or score_test - fea_5004 < -0.01:
        #sys.stderr.write('Error at line %d: hidden_qt-"%s" score is different, %.6f-%.6f\n'%(line_num, hidden_qt, fea_5004, score_test))
        #sys.stderr.write('\n')
        fail_count += 1
      #else:
      #  print qd + '\t' + hidden_qt + '\t' + str(fea_5004) + '\t' + str(score_test)
      total_count += 1
      qt = hidden_qt.split('\t')
      #print 'originalQueryString:%s\tdocid:%s\tquery_termid:%s\tdnn_title:%s\tcnn_title:%s\t6004:%.6f'%(query_str,doc_id,qt[0],qt[1])
      #sys.stdout.write('originalQueryString:%s\tdocid:%s\t%s\t6004:%.6f\n'%(query_str,doc_id,qd_to_qdc[qd],score_test))
      #sys.stderr.write('originalQueryString:%s\tdocid:%s\t6004:%.6f\n'%(query_str,doc_id,score_test))
    else:
      sys.stderr.write('originalQueryString:%s\tdocid:%s is not in the cache !!!\n'%(query_str,doc_id))
    #else:
    #  sys.stderr.write('Error at line %d: qd-"%s" is not in the cache !\n'%(line_num, qd))
  sys.stderr.write('line_num=%d\n'%line_num)
  sys.stderr.write("right/total=%d/%d=%.5f\n"%(total_count-fail_count, total_count,float(total_count-fail_count)/float(total_count)))

  ifs.close()
  #for k,v in hidden_to_qt.items():
  #  print v

