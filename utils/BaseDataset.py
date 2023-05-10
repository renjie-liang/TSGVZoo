
import torch
import numpy as np
from utils.utils import iou_n1, frac_idx
import torch.nn.functional as F  
from utils.data_utils import pad_seq, pad_char_seq, pad_video_seq
from models.BaseLib.layers import convert_length_to_mask
from utils.data_utils import video_augmentation, sample_vfeat_linear, label_idx



def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero(as_tuple=False)
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores



class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, video_features, configs, loadertype):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        # self.dataset.sort(key=lambda x:x['vid'])

        self.video_features = video_features
        self.max_vlen =  configs.model.max_vlen
        # self.truncate = configs.dataprocess.truncate
        # self.truncate_range = configs.dataprocess.truncate_range
        self.aug = configs.dataprocess.video_augmentation
        self.label_threshold = configs.dataprocess.label_threshold

        self.sample_type = configs.dataprocess.sample_type
        self.loadertype = loadertype

    def __getitem__(self, index):
        index = index
        record = self.dataset[index]
        vfeat = self.video_features[record['vid']]
        words_id, chars_id = record['wids'], record['cids']

        sfrac, efrac = record["se_frac"]
        # sidx, eidx = frac_idx([sfrac, efrac], vfeat.shape[0])

        ## ---- video augmentation
        vfeat, label_ = video_augmentation(sfrac, efrac, vfeat, aug=self.aug)
        assert not torch.all(label_ == 0), "in video augmentation: {}".format(record)
        vfeat, label = sample_vfeat_linear(vfeat, label_, self.max_vlen, self.sample_type)
        assert not torch.all(label == 0), "in video sampling: {} {} {}".format(record, label, label_)
        sidx, eidx = label_idx(label)

        ### !!! need check
        label_1D = self.gene_label_1D(sidx, eidx)
        # vfeat, _ = sample_vfeat_linear(vfeat, label_1D, self.max_vlen, self.sample_type)
        NER_label = self.gene_label_NER(sidx, eidx, vfeat)
        label_2D = self.gene_label_2D(sfrac, efrac, 1.0)
        # label_2D = self.gene_label_2D(record['s_time'], record['e_time'], record['duration'])

        # bert_id, bert_tmask = record["bert_id"], record["bert_mask"]
        # map2d_contrasts = self.get_map2d_contrast(sidx, eidx)
        # label_1D_t0 = self.load_label_1D_teach(self.logits_t0, index, record['vid'], vfeat.shape[0])

        res = {"record": record,
               "vid": record['vid'], 
               "max_vlen": self.max_vlen,
               "vfeat": vfeat,
               "words_id": words_id,
               "chars_id": chars_id,
               "label_1D": label_1D,
               "label_2D": label_2D,
               "NER_label": NER_label,
            #    "map2d_contrast": map2d_contrasts,
               "se_time": record["se_time"],
               "se_frac": [sfrac, efrac],
               "se_idx": [sidx, eidx],
            }
        return res

    def __len__(self):
        return len(self.dataset)

    def gene_label_1D(self, sidx, eidx):
        visual_len = self.max_vlen
        dist_idx = np.zeros((2, visual_len), dtype=np.float32)
        gt_s, gt_e = sidx, eidx
        gt_length = gt_e - gt_s + 1  # make sure length > 0
        dist_idx[0, :] = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_s) / (0.1 * gt_length)))
        dist_idx[1, :] = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_e) / (0.1 * gt_length)))
        dist_idx[0, dist_idx[0, :] >= 0.8] = 1.
        dist_idx[0, dist_idx[0, :] < 0.1353] = 0.
        dist_idx[1, dist_idx[1, :] >= 0.8] = 1.
        dist_idx[1, dist_idx[1, :] < 0.1353] = 0.
        if (dist_idx[0, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_s) / (0.1 * gt_length)))
            idx = np.argsort(p)
            dist_idx[0, idx[-1]] = 1.
        if (dist_idx[1, :] > 0.4).sum() == 0:
            p = np.exp(-0.5 * np.square((np.arange(visual_len) - gt_e) / (0.1 * gt_length)))
            idx = np.argsort(p)
            dist_idx[1, idx[-1]] = 1.
        dist_idx = torch.from_numpy(dist_idx)
        return dist_idx

    def gene_label_NER(self, sidx, eidx, vfeat):
        max_len = self.max_vlen
        cur_max_len = len(vfeat)
        st, et = sidx, eidx
        NER_label = torch.zeros([max_len], dtype=torch.int64) 

        ext_len = 1
        new_st_l = max(0, st - ext_len)
        new_st_r = min(st + ext_len, cur_max_len - 1)
        new_et_l = max(0, et - ext_len)
        new_et_r = min(et + ext_len, cur_max_len - 1)
        if new_st_r >= new_et_l:
            new_st_r = max(st, new_et_l - 1)
        NER_label[new_st_l:(new_st_r + 1)] = 1  # add B-M labels
        NER_label[(new_st_r + 1):new_et_l] = 2  # add I-M labels
        NER_label[new_et_l:(new_et_r + 1)] = 3  # add E-M labels

        return NER_label
    
    def gene_label_2D(self, stime, etime, duration):
        num_clips = self.max_vlen
        moment = torch.as_tensor([stime, etime])
        iou2d = torch.ones(num_clips, num_clips)
        candidates, _ = score2d_to_moments_scores(iou2d, num_clips, duration)
        iou2d = iou_n1(candidates, moment).reshape(num_clips, num_clips)
        return iou2d


class BaseCollate():
    def __init__(self):
        pass

    def __call__(self, datas):
        records, se_times, se_fracs = [], [], []
        vfeats, words_ids, chars_ids = [], [], []
        label_1Ds, label_2Ds, NER_labels = [], [], []
        max_vlen = datas[0]["max_vlen"]
        for d in datas:
            records.append(d["record"])
            vfeats.append(d["vfeat"])
            words_ids.append(d["words_id"])
            label_1Ds.append(d["label_1D"])
            label_2Ds.append(d["label_2D"])
            se_times.append(d["se_time"])
            se_fracs.append(d["se_frac"])
            chars_ids.append(d["chars_id"])
            NER_labels.append(d['NER_label'])
        # process text
        words_ids, _ = pad_seq(words_ids)
        words_ids = torch.as_tensor(words_ids, dtype=torch.int64)
        tmasks = (torch.zeros_like(words_ids) != words_ids).float()
        
        chars_ids, _ = pad_char_seq(chars_ids)
        chars_ids = torch.as_tensor(chars_ids, dtype=torch.int64)

        # process video 
        vfeats, vlens = pad_video_seq(vfeats, max_vlen)
        vfeats = torch.stack(vfeats)
        vlens = torch.as_tensor(vlens, dtype=torch.int64)
        vmasks = convert_length_to_mask(vlens, max_len=max_vlen)
        
        # process label
        label_1Ds = torch.stack(label_1Ds)
        label_2Ds = torch.stack(label_2Ds)
        NER_labels = torch.stack(NER_labels)
        
        se_times = torch.as_tensor(se_times, dtype=torch.float)
        se_fracs = torch.as_tensor(se_fracs, dtype=torch.float)

        res = {'words_ids': words_ids,
                'char_ids': chars_ids,
                'tmasks': tmasks,

                'vfeats': vfeats,
                'vmasks': vmasks,

                'label_1Ds': label_1Ds,
                'label_2Ds': label_2Ds,
                'NER_labels': NER_labels,

                # evaluate
                'se_times': se_times,
                'se_fracs': se_fracs,
                }

        return res, records