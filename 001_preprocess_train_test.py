import pandas as pd
import numpy as np

import gc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from utils import floorify_frac, floorify_ones_and_zeros, floorify
from config import config

def denoise(x):
    x['B_2'] = floorify_frac(x['B_2'],1/100)
    #B_3 - Exponential random variable
    x['B_4'] = floorify_frac(x['B_4'],1/78)
    #B_5 - Exponential random variable
    #B_6 - Exponential random variable
    #B_7 - Exponential random variable
    x['B_8'] = floorify_frac(x['B_8'],1)
    #B_9 - Joint distribution random variable
    #B_10 - Exponential random variable with limit
    #B_11 - Exponential random variable with limits
    #B_12 - Exponential random variable
    #B_13 - Exponential random variable
    #B_14 - Exponential random variable
    #B_15 - Exponential random variable
    x['B_16'] = floorify_frac(x['B_16'],1/12)
    x['B_17'] = np.clip(x['B_17'], -999, 1)
    x['B_18'] = np.clip(x['B_18'], -999, 1) # Seems to be an integer but need to continue to investigate
    # can be rounded up as each bin has AUC of 0.5
    x['B_19'] = np.floor(x['B_19']*100).fillna(-1).astype(np.int8)
    x['B_20'] = floorify_frac(x['B_20'],1/17)
    #B_21 - Confused on the distribution. Most values are 0 with noise added. Could floor values 0 < x < 0.01 to 0
    x['B_22'] = floorify_frac(x['B_22'],1/2)
    #B_23 - Exponential random variable
    #B_24 - Confused
    #B_25 - Exponential random variable wih peak at 0 and values < 0. Maybe clip this at 0
    #B_26 - Could floor values 0 < x < 0.01 to 0
    #B_27 - Could floor values 0 < x < 0.01 to 0
    #B_28 - Exponential random variable
    #B_29 - Could floor values 0 < x < 0.01 to 0
    x['B_30'] = floorify_frac(x['B_30'])
    x['B_31'] = floorify_frac(x['B_31'])
    x['B_32'] = floorify_frac(x['B_32'])
    x['B_33'] = floorify_frac(x['B_33'])
    #B_36 - Could floor values 0 < x < 0.01 to 0
    #B_37 - Confused
    x['B_38'] = floorify_frac(x['B_38'])
    x['B_39'] = np.clip(x['B_39'], -999, 1)
    #B_40 - Continuous Variable
    x['B_41'] = floorify_frac(x['B_41'])
    #B_42 - Continuous Variable


    x['D_39'] = floorify_frac(x['D_39'],1/34)
    #D_41 - Could floor values 0 < x < 0.01 to 0
    #D_42 - Exponential random variable
    #D_43 - Joint distribution random variable
    x['D_44'] = floorify_frac(x['D_44'],1/8)
    #D_45 - Joint distribution random variable
    #D_46 - Normal random variable
    #D_47 - Joint distribution random variable
    #D_48 - Joint distribution random variable
    x['D_49'] = floorify_frac(x['D_49'],1/71)
    x['D_51'] = floorify_frac(x['D_51'],1/3)
    x['D_52'] = np.clip(x['D_52'], -999, 1)
    x['D_54'] = np.clip(x['D_54'], -999, 1)
    #D_58 - Could floor values 0 < x < 0.01 to 0
    x['D_59'] = floorify_frac(x['D_59']+5/48,1/48)
    x['D_60'] = np.clip(x['D_60'], -999, 1)
    x['D_65'] = floorify_frac(x['D_65'],1/38)
    x['D_66'] = floorify_frac(x['D_66'])
    x['D_68'] = floorify_frac(x['D_68'])
    x['D_70'] = floorify_frac(x['D_70'],1/4)
    x['D_72'] = floorify_frac(x['D_72'],1/3)
    x['D_74'] = floorify_frac(x['D_74'],1/14)
    x['D_75'] = floorify_frac(x['D_75'],1/15)
    x['D_77'] = np.clip(x['D_77'], -999, 4)
    x['D_78'] = floorify_frac(x['D_78'],1/2)
    x['D_79'] = floorify_frac(x['D_79'],1/2)
    x['D_80'] = floorify_frac(x['D_80'],1/5)
    x['D_81'] = floorify_frac(x['D_81'])
    x['D_82'] = floorify_frac(x['D_82'],1/2)
    x['D_83'] = floorify_frac(x['D_83'])
    x['D_84'] = floorify_frac(x['D_84'],1/2)
    x['D_86'] = floorify_frac(x['D_86'])
    x['D_87'] = floorify_frac(x['D_87'])
    x['D_89'] = floorify_frac(x['D_89'],1/9)
    x['D_91'] = floorify_frac(x['D_91'],1/2)
    x['D_92'] = floorify_frac(x['D_92'])
    x['D_93'] = floorify_frac(x['D_93'])
    x['D_94'] = floorify_frac(x['D_94'])
    x['D_96'] = floorify_frac(x['D_96'])
    x['D_102'] = np.clip(x['D_102'], -999, 1.08)
    x['D_103'] = floorify_frac(x['D_103'])
    x.loc[x['D_104'] <= 0.01,'D_104'] = 0
    x['D_106'] = floorify_frac(x['D_106'],1/23)
    x['D_107'] = floorify_frac(x['D_107'],1/3)
    x['D_108'] = floorify_frac(x['D_108'])
    x['D_109'] = floorify_frac(x['D_109'])
    x['D_110'] = np.clip(x['D_110'], -999, 1)
    x['D_111'] = floorify_frac(x['D_111'],1/2)
    x['D_112'] = np.clip(x['D_112'], -999, 1)
    x['D_113'] = floorify_frac(x['D_113'],1/5)
    x['D_114'] = floorify_frac(x['D_114'])
    x['D_116'] = floorify_frac(x['D_116'])
    x['D_117'] = floorify_frac(x['D_117']+1)
    x['D_120'] = floorify_frac(x['D_120'])
    x['D_122'] = floorify_frac(x['D_122'],1/7)
    x['D_123'] = floorify_frac(x['D_123'])
    x['D_124'] = floorify_frac(x['D_124']+1/22,1/22)
    x['D_125'] = floorify_frac(x['D_125'])
    x['D_126'] = floorify_frac(x['D_126']+1)
    x['D_127'] = floorify_frac(x['D_127'])
    x['D_129'] = floorify_frac(x['D_129'])
    x['D_130'] = floorify_frac(x['D_130'], 1/100)
    x['D_133'] = np.clip(x['D_133'], -999, 1.5)
    x.loc[x['D_133'] <= 0.01,'D_133'] = 0
    x['D_134'] = np.clip(x['D_134'], -999, 1)
    x['D_135'] = floorify_frac(x['D_135'])
    x['D_136'] = floorify_frac(x['D_136'],1/4)
    x['D_137'] = floorify_frac(x['D_137'])
    x['D_138'] = floorify_frac(x['D_138'],1/2)
    x['D_139'] = floorify_frac(x['D_139'])
    x['D_140'] = floorify_frac(x['D_140'])
    x.loc[x['D_141'] <= 0.01,'D_141'] = 0
    x['D_143'] = floorify_frac(x['D_143'])
    x.loc[x['D_144'] <= 0.01,'D_144'] = 0
    x['D_145'] = floorify_frac(x['D_145'],1/11)

    x.loc[x['R_1'] <= 0.01,'R_1'] = 0 #Can do more here

    x['R_2'] = floorify_frac(x['R_2'])
    x['R_3'] = floorify_frac(x['R_3'],1/10)
    x['R_4'] = floorify_frac(x['R_4'])
    x['R_5'] = floorify_frac(x['R_5'],1/2)
    x['R_8'] = floorify_frac(x['R_8'])
    x['R_9'] = floorify_frac(x['R_9'],1/6)
    x['R_10'] = floorify_frac(x['R_10'])
    x['R_11'] = floorify_frac(x['R_11'],1/2)
    x['R_12'] = np.clip(x['R_12'], -999, 1)
    x['R_13'] = floorify_frac(x['R_13'],1/31)
    x['R_15'] = floorify_frac(x['R_15'])
    x['R_16'] = floorify_frac(x['R_16'],1/2)
    x['R_17'] = floorify_frac(x['R_17'],1/35)
    x['R_18'] = floorify_frac(x['R_18'],1/31)
    x['R_19'] = floorify_frac(x['R_19'])
    x['R_20'] = floorify_frac(x['R_20'])
    x['R_21'] = floorify_frac(x['R_21'])
    x['R_22'] = floorify_frac(x['R_22'])
    x['R_23'] = floorify_frac(x['R_23'])
    x['R_24'] = floorify_frac(x['R_24'])
    x['R_25'] = floorify_frac(x['R_25'])
    x['R_26'] = floorify_frac(x['R_26'],1/28)
    x['R_27'] = np.clip(x['R_27'], -999, 1)
    x['R_28'] = floorify_frac(x['R_28'])

    x['S_6'] = floorify_frac(x['S_6'])
    x['S_11'] = floorify_frac(x['S_11']+5/25,1/25)
    #S_13 Couldn't figure the distance

    x['S_15'] = floorify_frac(x['S_15']+3/10,1/10)
    x.loc[x['S_16'] <= 0.01,'S_16'] = 0
    x.loc[x['S_17'] <= 0.01,'S_17'] = 0
    x['S_18'] = floorify_frac(x['S_18'])
    x.loc[x['S_19'] <= 0.01,'S_19'] = 0
    x['S_20'] = floorify_frac(x['S_20'])

    x['P_2'] = np.clip(x['P_2'], -999, 1)
    x.loc[x['P_4'] <= 0.01,'P_4'] = 0

    #Categorical casting
    x['D_63'] = x['D_63'].apply(lambda t: {'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}[t]).astype(np.int8)
    x['D_64'] = x['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)

    # one value overlaps, but the split can identified by S_11
    x.loc[x.S_13.between(0.67, 0.7) & (x.S_11.isin([15,16,17])),'S_13'] = 0.6789168283158535
    floor_vals = (0, 0.0377176456223467, 0.2804642206328049, 0.4013539714415651, 0.4206963381303189, 0.5067698438641042,
                  0.5261121975338173, 0.5551258157960416, 0.6218568673028206, 0.6876208933830246, 0.8433269036807703, 1)
    for c in floor_vals:
        x['S_13'] = x['S_13'].apply(lambda t: floorify(t,c))
    x['S_13'] = np.round(x['S_13']*1034).fillna(-1).astype(np.int16)

    # this one has many more value overlaps, but the splits can be identified by S_15
    x.loc[(x.S_8>=0.30) & (x.S_8<=0.35) & (x.S_15<=6),'S_8'] = 0.3224889650033656
    x.loc[(x.S_8>=0.30) & (x.S_8<=0.35) & (x.S_15==7),'S_8'] = 0.3145925513763017
    x.loc[(x.S_8>=0.45) & (x.S_8<=0.477) & (x.S_15==3),'S_8'] = 0.4570436553944634
    x.loc[(x.S_8>=0.45) & (x.S_8<=0.477) & (x.S_15==5),'S_8'] = 0.4636765662005172
    x.loc[(x.S_8>=0.45) & (x.S_8<=0.477) & (x.S_15==6),'S_8'] = 0.4592546209653157
    x.loc[(x.S_8>=0.55) & (x.S_8<=0.65) & (x.S_15==5),'S_8'] = 0.5938092592144236
    x.loc[(x.S_8>=0.55) & (x.S_8<=0.65) & (x.S_15==4),'S_8'] = 0.5994946974629933
    x.loc[(x.S_8>=0.55) & (x.S_8<=0.65) & (x.S_15<=2),'S_8'] = 0.6017056828901041
    x.loc[(x.S_8>=0.73) & (x.S_8<=0.78) & (x.S_15==3),'S_8'] = 0.7441567340107059
    x.loc[(x.S_8>=0.73) & (x.S_8<=0.78) & (x.S_15==5),'S_8'] = 0.7517372106519937
    x.loc[(x.S_8>=0.73) & (x.S_8<=0.78) & (x.S_15==4),'S_8'] = 0.7586861099807893
    x.loc[(x.S_8>=0.91) & (x.S_8<=0.98) & (x.S_15==4),'S_8'] = 0.9147189165383852
    x.loc[(x.S_8>=0.91) & (x.S_8<=0.98) & (x.S_15<=2),'S_8'] = 0.9327230426634736
    x.loc[(x.S_8>=0.91) & (x.S_8<=0.98) & (x.S_15==3),'S_8'] = 0.935565546481781
    x.loc[(x.S_8>=1.12) & (x.S_8<=1.17) & (x.S_15<=2),'S_8'] = 1.1440303975988897
    x.loc[(x.S_8>=1.12) & (x.S_8<=1.17) & (x.S_15==3),'S_8'] = 1.151926881019957
    floor_vals = (0, 0.1017056275625063, 0.119709415455368, 0.1667719530078215, 0.2438408100936861,
                  0.3578648754166172, 0.4055590769093041, 0.4772583808904347, 0.4876816287061991,
                  0.6620341135675392, 0.7005685574395781, 0.8509160456526623, 1, 1.0145299163657109,
                  1.1051803467580654, 1.2214158871037435)
    for c in floor_vals:
        x['S_8'] = x['S_8'].apply(lambda t: floorify(t,c))
    x['S_8'] = np.round(x['S_8']*3166).fillna(-1).astype(np.int16)

    cols = x.select_dtypes(include=[float]).columns
    for col in tqdm(cols):
        x[col] = floorify_ones_and_zeros(x[col])

    for col in x.select_dtypes(include=[float]).columns.tolist():
        x[col] = x[col].astype(np.float32)

    x.drop(config.dropcols, axis=1, inplace=True)

    return x

def main():
    train = pd.read_csv("input/train_data.csv")
    test = pd.read_csv("input/test_data.csv")

    ntrain = len(train)
    df = pd.concat([train, test], axis=0, ignore_index=True)
    del train, test
    gc.collect()

    df = denoise(df)
    df[:ntrain].reset_index(drop=True).to_feather("train_processed/train.f")
    df[ntrain:].reset_index(drop=True).to_feather("test_processed/test.f")
    del df
    gc.collect()

if(__name__ == '__main__'):
    main()