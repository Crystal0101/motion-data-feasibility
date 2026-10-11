import unittest
import numpy as np
from src.corrected_data import transform,frame_features,block_features,pooled_features
from src.corrected_private import gradients,clip
from src.corrected_models import scores

class CorrectedChecks(unittest.TestCase):
    def test_transform_preserves_order(self):
        points=np.linspace(11.2,34.8,100)
        self.assertTrue((np.diff(transform(points))>0).all())
        self.assertTrue((np.abs(transform(np.array([-1e12,0,1e12])))<=1).all())
    def test_features_no_c4_collapse(self):
        a=np.tile(np.array([1.,2.,3.,12.,.604]),(10,1))
        b=a.copy();b[:,3]=30
        self.assertNotEqual(frame_features(a)[3],frame_features(b)[3])
    def test_empty_payload_vs_absent(self):
        self.assertEqual(frame_features(np.zeros((0,5)))[-1],1)
        self.assertEqual(frame_features(np.zeros((0,5)))[:-1].sum(),0)
    def test_gap_not_concatenated(self):
        frames=np.zeros((297,27));frames[0]=1;frames[20]=.5;frames[20,-1]=1
        blocks=block_features(frames)
        self.assertEqual(blocks.shape,(30,27))
        np.testing.assert_array_equal(blocks[1],np.zeros(27))
        self.assertAlmostEqual(float(blocks[0,-1]),.1,places=6)
        self.assertAlmostEqual(float(blocks[2,0]),.5,places=6)
    def test_last_block_seven_slots(self):
        frames=np.zeros((297,27));frames[296]=1
        self.assertAlmostEqual(float(block_features(frames)[-1,-1]),1/7,places=6)
    def test_pool_dimensions(self):
        self.assertEqual(pooled_features(np.zeros((3,30,27))).shape,(3,108))
    def test_subject_gradient_finite_difference(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(5,2));y=np.array([0,1,0,1,1])
        owners=np.array(['a','a','b','b','b']);p=rng.normal(size=(3,2))
        actual=gradients(x,y,owners,p)
        for subject_id,subject in enumerate(['a','b']):
            mask=owners==subject
            def loss(par):
                logits=x[mask]@par[:-1]+par[-1]
                logs=logits-np.logaddexp.reduce(logits,axis=1,keepdims=True)
                return -logs[np.arange(mask.sum()),y[mask]].mean()
            for index in np.ndindex(p.shape):
                a,b=p.copy(),p.copy();a[index]+=1e-6;b[index]-=1e-6
                self.assertAlmostEqual(actual[subject_id][index],(loss(a)-loss(b))/2e-6,places=7)
    def test_joint_clip(self):
        g=np.arange(24.).reshape(2,3,4)
        self.assertTrue((np.linalg.norm(clip(g),axis=(1,2))<=1+1e-12).all())
        self.assertEqual(clip(g[:0]).sum(0).shape,(3,4))
    def test_subject_balanced_confusion(self):
        actual=scores(np.r_[0,np.ones(99,dtype=int)],np.zeros(100,dtype=int),np.array(['a']+['b']*99))
        self.assertAlmostEqual(actual['accuracy'],.5)
        self.assertAlmostEqual(actual['macro_f1'],(2/3)/27)

if __name__=='__main__':unittest.main()
