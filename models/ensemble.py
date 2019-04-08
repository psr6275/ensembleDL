import torch.nn as nn


class Correct(nn.Module):
    def __init__(self,base_clf,dropout_tf = False):
        super(Correct, self).__init__()
        self.num_features,self.features = self._get_conv_layers(base_clf)
        self.correctness = nn.Linear(self.num_features,1)
        self.dropout_tf = dropout_tf
        if dropout_tf:
            self.dropout = nn.Dropout2d(0.5)
        self.prob = nn.Sigmoid()

    def forward(self,x):
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.correctness(out)
        if self.dropout_tf:
            out = self.dropout(out)
        out = self.prob(out)
        return out

    def _get_conv_layers(self,clf):
        for i,m in enumerate(clf.modules()):
            if isinstance(m,nn.Linear):
                num_features = m.in_features
                break
        return num_features,nn.Sequential(*list(clf.features.children())[:i])


class Ensemble(nn.Module):
    def __init__(self,model_list):
        super(Ensemble, self).__init__()

