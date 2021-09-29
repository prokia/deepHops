#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    UpdateFunction.py: Updates the nodes using the previous state and the message.
    
    Usage:

"""

from __future__ import print_function

# Own modules

from .MessageFunction import MessageFunction
from .nnet import NNet

import numpy as np
import time
import os
import argparse
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable

#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor


class UpdateFunction(nn.Module):

    # Constructor
    def __init__(self, update_def='nn', args={}):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.__set_update(update_def, args)

    # Update node hv given message mv
    def forward(self, h_v, m_v, opt={}):
        return self.u_function(h_v, m_v, opt)

    # Set update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'duvenaud':         self.u_duvenaud,            
                    'intnet':             self.u_intnet,
                    'mpnn':             self.u_mpnn
                }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'duvenaud':         self.init_duvenaud,            
            'intnet':             self.init_intnet,
            'mpnn':             self.init_mpnn
        }.get(self.u_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    # Get the update function arguments
    def get_args(self):
        return self.args
    
    # Duvenaud
    def u_duvenaud(self, h_v, m_v, opt):

        param_sz = self.learn_args[0][opt['deg']].size()
        parameter_mat = torch.t(self.learn_args[0][opt['deg']])[None, ...].expand(m_v.size(0), param_sz[1], param_sz[0])
        
        #print(parameter_mat.size())
        #print(m_v.size())
        #print(torch.transpose(m_v.unsqueeze(-2), 1, 2).size())

        #aux = torch.bmm(parameter_mat, torch.transpose(m_v, 1, 2))
        aux = torch.bmm(parameter_mat, torch.transpose(m_v.unsqueeze(-2), 1, 2))

        return torch.transpose(torch.nn.Sigmoid()(aux), 1, 2)

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        # Filter degree 0 (the message will be 0 and therefore there is no update
        args['deg'] = [i for i in params['deg'] if i!=0]
        args['in'] = params['in']
        args['out'] = params['out']

        # Define a parameter matrix H for each degree.
        learn_args.append(torch.nn.Parameter(torch.randn(len(args['deg']), args['in'], args['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args    

    # Battaglia et al. (2016), Interaction Networks
    def u_intnet(self, h_v, m_v, opt):
        if opt['x_v'].ndimension():
            input_tensor = torch.cat([h_v, opt['x_v'], torch.squeeze(m_v)], 1)
        else:
            input_tensor = torch.cat([h_v, torch.squeeze(m_v)], 1)

        return self.learn_modules[0](input_tensor)

    def init_intnet(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in'] = params['in']
        args['out'] = params['out']

        learn_modules.append(NNet(n_in=params['in'], n_out=params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

    def u_mpnn(self, h_v, m_v, opt={}):
        h_in = h_v.view(-1,h_v.size(2))
     
        m_in = m_v.view(-1,m_v.size(2))
           
        h_new = self.learn_modules[0](m_in[None,...],h_in[None,...])[0] # 0 or 1???
        return torch.squeeze(h_new).view(h_v.size())

    def init_mpnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['in_m'] = params['in_m']
        args['out'] = params['out']

        # GRU
        learn_modules.append(nn.GRU(params['in_m'], params['out']))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args
