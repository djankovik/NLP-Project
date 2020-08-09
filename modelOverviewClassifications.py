from stanceDetectionModel_build_train_evaluate import *
from readDataFromFile import stances_onehot_test
import zipfile

def model_summary_evaluation_per_targets():
    print("model_summary_evaluation")
    leng = len(stances_onehot_test)
    if len(predictions_lemm_oh_t) != leng or len(predictions_lemm_vec_t) != leng or len(predictions_stem_oh_t) != leng or len(predictions_stem_vec_t) != leng or len(predictions_raw_t) != leng or len(predictions_fltr_t) != leng or len(predictions_stops_t) != leng or len(predictions_afin_t) != leng or len(predictions_afinsum_t) != leng or len(predictions_nrcaffin_t) != leng or len(predictions_nrcaffin_sum_t) != leng or len(predictions_nrcvad_t) != leng or len(predictions_nrcvadsum_t) != leng or len(predictions_sia_t) != leng or len(predictions_iar_t) != leng or len(predictions_iarfilt_t) != leng or len(predictions_iarstop_t) != leng or len(predictions_pno_t) != leng or len(predictions_pno_summary_t) != leng or len(predictions_bi_t) != leng or len(predictions_bi_specvec_t) != leng or len(predictions_bi_vec_t) != leng or len(predictions_four_t) != leng or len(predictions_four_specvec_t) != leng or len(predictions_four_vec_t) != leng or len(predictions_tri_t) != leng or len(predictions_tri_specvec_t) != leng or len(predictions_tri_vec_t) != leng or len(predictions_uni_t) != leng or len(predictions_uni_specvec_t) != leng or len(predictions_uni_vec_t) != leng or len(predictions_hashtags_oh_t) != leng or len(predictions_hashtags_vec_t) != leng:
        print("!!!NOT ALL PREDICTIONS WERE OF THE EXPECTED LENGTH!!!")
        return

    predictions = []
    votes_summary = []
    correct_majority_votes_cnt = 0
    stats = {0:{0:0,1:0,2:0},1:{0:0,1:0,2:0},2:{0:0,1:0,2:0}}   

    for (expected,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33) in zip(stances_onehot_test, predictions_lemm_oh_t,predictions_lemm_vec_t,predictions_stem_oh_t,predictions_stem_vec_t,predictions_raw_t,predictions_fltr_t,predictions_stops_t,predictions_afin_t,predictions_afinsum_t,predictions_nrcaffin_t,predictions_nrcaffin_sum_t,predictions_nrcvad_t,predictions_nrcvadsum_t,predictions_sia_t,predictions_iar_t,predictions_iarfilt_t,predictions_iarstop_t,predictions_pno_t,predictions_pno_summary_t,predictions_bi_t,predictions_bi_specvec_t,predictions_bi_vec_t,predictions_four_t,predictions_four_specvec_t,predictions_four_vec_t,predictions_tri_t,predictions_tri_specvec_t,predictions_tri_vec_t,predictions_uni_t,predictions_uni_specvec_t,predictions_uni_vec_t,predictions_hashtags_oh_t,predictions_hashtags_vec_t):
        exp = expected.index(max(expected))
        pred1 = p1.tolist().index(max(p1.tolist()))
        pred2 = p2.tolist().index(max(p2.tolist()))
        pred3= p3.tolist().index(max(p3.tolist()))
        pred4 = p4.tolist().index(max(p4.tolist()))
        pred5 = p5.tolist().index(max(p5.tolist()))
        pred6 = p6.tolist().index(max(p6.tolist()))
        pred7 = p7.tolist().index(max(p7.tolist()))
        pred8 = p8.tolist().index(max(p8.tolist()))
        pred9 = p9.tolist().index(max(p9.tolist()))
        pred10 = p10.tolist().index(max(p10.tolist()))
        pred11 = p11.tolist().index(max(p11.tolist()))
        pred12 = p12.tolist().index(max(p12.tolist()))
        pred13= p13.tolist().index(max(p13.tolist()))
        pred14 = p14.tolist().index(max(p14.tolist()))
        pred15 = p15.tolist().index(max(p15.tolist()))
        pred16 = p16.tolist().index(max(p16.tolist()))
        pred17 = p17.tolist().index(max(p17.tolist()))
        pred18 = p18.tolist().index(max(p18.tolist()))
        pred19 = p19.tolist().index(max(p19.tolist()))
        pred20 = p20.tolist().index(max(p20.tolist()))
        pred21 = p21.tolist().index(max(p21.tolist()))
        pred22 = p22.tolist().index(max(p22.tolist()))
        pred23= p23.tolist().index(max(p23.tolist()))
        pred24 = p24.tolist().index(max(p24.tolist()))
        pred25 = p25.tolist().index(max(p25.tolist()))
        pred26 = p26.tolist().index(max(p26.tolist()))
        pred27 = p27.tolist().index(max(p27.tolist()))
        pred28 = p28.tolist().index(max(p28.tolist()))
        pred29 = p29.tolist().index(max(p29.tolist()))
        pred30 = p30.tolist().index(max(p30.tolist()))
        pred31 = p31.tolist().index(max(p31.tolist()))
        pred32 = p32.tolist().index(max(p32.tolist()))
        pred33 = p33.tolist().index(max(p33.tolist()))

         
        cnt = {0:0,1:0,2:0} #0=NONE, 1=FAVOR, 2=AGAINST

        for vote in [pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33]:
            cnt[vote]=cnt[vote]+1
            stats[exp][vote] = stats[exp][vote]+1

        sortedcnt = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1],reverse=True)}
        if exp == list(sortedcnt.keys())[0]:
            correct_majority_votes_cnt+=1

        votes_summary.append([exp,cnt])
        predictions.append([exp,[pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33]])
        
    return (correct_majority_votes_cnt,stats,predictions,votes_summary)

def model_summary_evaluation():
    print("model_summary_evaluation")
    leng = len(stances_onehot_test)
    if len(predictions_lemm_oh) != leng or len(predictions_lemm_vec) != leng or len(predictions_stem_oh) != leng or len(predictions_stem_vec) != leng or len(predictions_raw) != leng or len(predictions_fltr) != leng or len(predictions_stops) != leng or len(predictions_afin) != leng or len(predictions_afinsum) != leng or len(predictions_nrcaffin) != leng or len(predictions_nrcaffin_sum) != leng or len(predictions_nrcvad) != leng or len(predictions_nrcvadsum) != leng or len(predictions_sia) != leng or len(predictions_iar) != leng or len(predictions_iarfilt) != leng or len(predictions_iarstop) != leng or len(predictions_pno) != leng or len(predictions_pno_summary) != leng or len(predictions_bi) != leng or len(predictions_bi_specvec) != leng or len(predictions_bi_vec) != leng or len(predictions_four) != leng or len(predictions_four_specvec) != leng or len(predictions_four_vec) != leng or len(predictions_tri) != leng or len(predictions_tri_specvec) != leng or len(predictions_tri_vec) != leng or len(predictions_uni) != leng or len(predictions_uni_specvec) != leng or len(predictions_uni_vec) != leng or len(predictions_hashtags_oh) != leng or len(predictions_hashtags_vec) != leng:
        print("!!!NOT ALL PREDICTIONS WERE OF THE EXPECTED LENGTH!!!")
        return

    predictions = []
    votes_summary = []
    correct_majority_votes_cnt = 0
    stats = {0:{0:0,1:0,2:0},1:{0:0,1:0,2:0},2:{0:0,1:0,2:0}}   

    for (expected,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33) in zip(stances_onehot_test,predictions_lemm_oh,predictions_lemm_vec,predictions_stem_oh,predictions_stem_vec,predictions_raw,predictions_fltr,predictions_stops,predictions_afin,predictions_afinsum,predictions_nrcaffin,predictions_nrcaffin_sum,predictions_nrcvad,predictions_nrcvadsum,predictions_sia,predictions_iar,predictions_iarfilt,predictions_iarstop,predictions_pno,predictions_pno_summary,predictions_bi,predictions_bi_specvec,predictions_bi_vec,predictions_four,predictions_four_specvec,predictions_four_vec,predictions_tri,predictions_tri_specvec,predictions_tri_vec,predictions_uni,predictions_uni_specvec,predictions_uni_vec,predictions_hashtags_oh,predictions_hashtags_vec):
        exp = expected.index(max(expected))
        pred1 = p1.tolist().index(max(p1.tolist()))
        pred2 = p2.tolist().index(max(p2.tolist()))
        pred3= p3.tolist().index(max(p3.tolist()))
        pred4 = p4.tolist().index(max(p4.tolist()))
        pred5 = p5.tolist().index(max(p5.tolist()))
        pred6 = p6.tolist().index(max(p6.tolist()))
        pred7 = p7.tolist().index(max(p7.tolist()))
        pred8 = p8.tolist().index(max(p8.tolist()))
        pred9 = p9.tolist().index(max(p9.tolist()))
        pred10 = p10.tolist().index(max(p10.tolist()))
        pred11 = p11.tolist().index(max(p11.tolist()))
        pred12 = p12.tolist().index(max(p12.tolist()))
        pred13= p13.tolist().index(max(p13.tolist()))
        pred14 = p14.tolist().index(max(p14.tolist()))
        pred15 = p15.tolist().index(max(p15.tolist()))
        pred16 = p16.tolist().index(max(p16.tolist()))
        pred17 = p17.tolist().index(max(p17.tolist()))
        pred18 = p18.tolist().index(max(p18.tolist()))
        pred19 = p19.tolist().index(max(p19.tolist()))
        pred20 = p20.tolist().index(max(p20.tolist()))
        pred21 = p21.tolist().index(max(p21.tolist()))
        pred22 = p22.tolist().index(max(p22.tolist()))
        pred23= p23.tolist().index(max(p23.tolist()))
        pred24 = p24.tolist().index(max(p24.tolist()))
        pred25 = p25.tolist().index(max(p25.tolist()))
        pred26 = p26.tolist().index(max(p26.tolist()))
        pred27 = p27.tolist().index(max(p27.tolist()))
        pred28 = p28.tolist().index(max(p28.tolist()))
        pred29 = p29.tolist().index(max(p29.tolist()))
        pred30 = p30.tolist().index(max(p30.tolist()))
        pred31 = p31.tolist().index(max(p31.tolist()))
        pred32 = p32.tolist().index(max(p32.tolist()))
        pred33 = p33.tolist().index(max(p33.tolist()))

         
        cnt = {0:0,1:0,2:0} #0=NONE, 1=FAVOR, 2=AGAINST

        for vote in [pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33]:
            cnt[vote]=cnt[vote]+1
            stats[exp][vote] = stats[exp][vote]+1

        sortedcnt = {k: v for k, v in sorted(cnt.items(), key=lambda item: item[1],reverse=True)}
        if exp == list(sortedcnt.keys())[0]:
            correct_majority_votes_cnt+=1

        votes_summary.append([exp,cnt])
        predictions.append([exp,[pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10,pred11,pred12,pred13,pred14,pred15,pred16,pred17,pred18,pred19,pred20,pred21,pred22,pred23,pred24,pred25,pred26,pred27,pred28,pred29,pred30,pred31,pred32,pred33]])
        
    return (correct_majority_votes_cnt,stats,predictions,votes_summary)

print("_______________________Model Summary - TARGETS NOT CONSIDERED in models____________________________")

correct_cnt,statistics,predictions,votes =  model_summary_evaluation()

print("Majority vote: correctly classified: "+str(correct_cnt)+" | Total: "+str(len(stances_onehot_test)))
print(statistics)

print("_______________________Model Summary - PER TARGET models____________________________")

correct_cnt_t,statistics_t,predictions_t,votes_t =  model_summary_evaluation_per_targets()

print("Majority vote_ correctly classified: "+str(correct_cnt_t)+" | Total: "+str(len(stances_onehot_test)))
print(statistics_t)