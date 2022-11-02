from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import time

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# So:
#  head_convs can be set to a list of output dimensions for each convolution
#  e.g. [256, 64, 64] which also determines the number of convolution blocks
#  We can also set the kernel size, just make sure to account for padding
#  We can also set whether to use ReLU at every layer or not

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, \
        actual_num_stacks, kernel_size, activations, head_conv_value, opt=None):

        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3


        self.num_stacks = num_stacks
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            head_conv = head_convs[head]

            head_conv = [head_conv_value for x in range(actual_num_stacks)]
            head_conv.insert(0, last_channel)

            new_kernel_size = [kernel_size for x in range(actual_num_stacks)]

            if len(head_conv) > 0:
              out = nn.Conv2d(head_conv[-1], classes,
                    kernel_size=1, stride=1, padding=0, bias=True)
              # conv = nn.Conv2d(last_channel, head_conv[0],
              #                  kernel_size=new_kernel_size[0],
              #                  padding=head_kernel // 2, bias=True)

              relu = nn.ReLU(inplace=True)
              modules = []

              for k in range(0, len(head_conv)-1):

                  modules.append(nn.Conv2d(head_conv[k], head_conv[k+1],
                               kernel_size=new_kernel_size[k], \
                               padding=new_kernel_size[k] // 2, bias=True))
                  if activations[k]:
                      modules.append(relu)

              # Add the final output layer
              modules.append(out)

              # Now stack the model together
              fc = nn.Sequential(*modules)


              # if len(convs) == 1:
              #   fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
              # elif len(convs) == 2:
              #   fc = nn.Sequential(
              #     convs[0], nn.ReLU(inplace=True),
              #     convs[1], nn.ReLU(inplace=True), out)
              # elif len(convs) == 3:
              #   fc = nn.Sequential(
              #       convs[0], nn.ReLU(inplace=True),
              #       convs[1], nn.ReLU(inplace=True),
              #       convs[2], nn.ReLU(inplace=True), out)
              # elif len(convs) == 4:
              #   fc = nn.Sequential(
              #       convs[0], nn.ReLU(inplace=True),
              #       convs[1], nn.ReLU(inplace=True),
              #       convs[2], nn.ReLU(inplace=True),
              #       convs[3], nn.ReLU(inplace=True), out)
              if 'hm' in head:
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2d(last_channel, classes,
                  kernel_size=1, stride=1, padding=0, bias=True)
              if 'hm' in head:
                fc.bias.data.fill_(opt.prior_bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)
        # asdf
        # time.sleep(10)

    def img2feats(self, x):
      raise NotImplementedError

    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def forward(self, x, pre_img=None, pre_hm=None):

      # print("ENTERING FORWARD BASE MODEL")
      # input = [y.shape for y in x]
      # print(input)
      # print(len(input))

      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
              z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks):
          z = {}
          for head in self.heads:
              z[head] = self.__getattr__(head)(feats[s])
          out.append(z)
      return out
