import sys
import os
import cherrypy
import numpy as np
import math


class retrievalserver(object):
    def __init__(self, param):
        self.mdl = param['simres']
        self.sim = []
        for f in param['simres']:
            self.sim += [np.load(f)]
        self.label = np.load(param['projlabel'])
        self.img_num = self.label.shape[0]
        print 'similarity matrix:', [_.shape for _ in self.sim]
        #print 'label matrix:', self.label

    def genVisPage(self, qid, rid_list, score_list, label_list):
        style = '@font-face{font-family: '+ '\'{}\'; src: url(\'/fonts/{}\');'.format('errfont', 'error.otf') + '}'
        style += """
                  table{
                  display: inline-block
                  }
                 """

        htmltxt = '<html><head><style>{}</style></head><body>'.format(style)
        qimg = 'images/img-{:05d}.jpg'.format(qid)
        htmltxt += '<img src=\"{}\"> id={}<br/><hr><br/>'.format(qimg, qid)
        htmltxt += '<table><tr>'
        for j in range(len(rid_list)):
            htmltxt += '<td align="center">{}</td>'.format(self.mdl[j])
        htmltxt += '</tr>'
        htmltxt += '<tr>'
        for j in range(len(rid_list)):
            htmltxt += '<td>'
            for i,idx in enumerate(rid_list[j]):
                img = 'images/img-{:05d}.jpg'.format(idx)
                s = score_list[j][i]
                l = label_list[j][i]
                if l == 1:
                    clr = '00EE00'
                else:
                    clr = 'white'
                tab = '<table><tr><td><img src=\"{}\"></td></tr><tr><td align="center", style="background-color:{}">{:.3f}</td></tr></table>'.format(img, clr, s)
                htmltxt += '<a href="/retrieve?qid={}">'.format(idx) + tab + '</a>'
            htmltxt += '</td>'
        htmltxt += '</tr></table>'
        htmltxt += '</body></html>'
        return htmltxt

    @cherrypy.expose
    def index(self):
        return """
        <html><body>
            <form action="retrieve" method="post" enctype="multipart/form-data">
            """ + \
            'Enter query image id (0 ~ {}):'.format(self.img_num-1) + \
            """
            <input type="text" value="123" name="qid" size="10" /> <br>
            <input type="submit" />
            <hr>
            </form>
        </body></html>
        """

    @cherrypy.expose
    def retrieve(self, qid):
        qid = int(qid)
        qid = max(0, min(qid, self.img_num-1))
        topn = 40

        idx_all = []
        score_all = []
        label_all = []
        for simmat in self.sim:
            score = simmat[qid]
            idx = np.argsort(score)
            idx = idx[::-1]
            #print sscore[0:10]
            idx = idx[1:topn+1]
            score = score[idx]
            label = self.label[qid, idx]
            idx_all += [idx]
            score_all += [score]
            label_all += [label]

        htmltxt = self.genVisPage(qid, idx_all, score_all, label_all)

        return htmltxt

if __name__ == '__main__':
    cherrypy.server.socket_host = '0.0.0.0'
    cherrypy.server.socket_port = 8010
    """
    param = {'simres': './result/project_verification_label_matrix.npy', 'projlabel': './result/project_verification_label_matrix.npy'}
    param = {'simres': ['./result/project_verification_label_matrix.npy', './result/Alexnet_Conv2_sim_matrix.npy'], \
             'projlabel': './result/project_verification_label_matrix.npy'}
    param = {'simres': ['./result/Alexnet_Conv2_sim_matrix.npy', './result/Train_FCN_sim_matrix.npy'], \
             'projlabel': './result/project_verification_label_matrix.npy'}
    
    param = {'simres': ['./result/Alexnet_sim_matrix.npy', './result/FCN+Train_Field+AlexConv5+StyleConv2.npy'], \
             'projlabel': './result/project_verification_label_matrix.npy'}
    """
    param = {'simres': ['./result/project_verification_label_matrix.npy', './result/FCN+Train_Field+AlexConv5+StyleConv2.npy'], \
             'projlabel': './result/project_verification_label_matrix.npy'}

    cherrypy.quickstart(retrievalserver(param), '/', 'resrc.conf')
