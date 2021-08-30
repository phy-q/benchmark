import numpy as np

from LearningAgents.RLNetwork.DQNBase import DQNBase
from StateReader.SymbolicStateReader import SymbolicStateReader


class DQNSymbolicBase(DQNBase):

    def __init__(self, h, w, outputs, if_save_local=False, writer=None, device='cpu'):
        super(DQNSymbolicBase, self).__init__(h=h, w=w, device=device, if_save_local=if_save_local, writer=writer,
                                              outputs=outputs)

        #####
        self.input_type = 'symbolic'
        self.output_type = 'discrete'
        self.model = np.loadtxt("Utils/model", delimiter=",")
        self.target_class = list(map(lambda x: x.replace("\n", ""), open('Utils/target_class').readlines()))
        #####

    def forward(self, x):
        return NotImplementedError()

    def transform_sparse(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image_sparse(h=self.h, w=self.w)

    def transform_full(self, state):
        return SymbolicStateReader(state, self.model, self.target_class).get_symbolic_image(h=self.h, w=self.w)

    def transform(self, state):
        return self.transform_full(state) if self.if_save_local else self.transform_sparse(state)
