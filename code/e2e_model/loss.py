"""
Hungarian Matching Loss for DETR-style discharge detection.

Matches predicted discharge queries to ground truth discharge times using
bipartite matching (Hungarian algorithm), then computes:
  - L1 loss on matched times
  - BCE loss on confidence (1 for matched, 0 for unmatched)
  - Huber loss on frequency prediction
  - BCE loss on laterality
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class HungarianMatchingLoss(torch.nn.Module):
    """Combined loss with Hungarian matching for discharge detection."""

    def __init__(self, w_time=5.0, w_conf=2.0, w_freq=1.0, w_lat=1.0,
                 lambda_no_obj=0.5):
        """
        Args:
            w_time: weight for time L1 loss
            w_conf: weight for confidence BCE loss
            w_freq: weight for frequency Huber loss
            w_lat: weight for laterality BCE loss
            lambda_no_obj: cost penalty for unmatched predictions in matching
        """
        super().__init__()
        self.w_time = w_time
        self.w_conf = w_conf
        self.w_freq = w_freq
        self.w_lat = w_lat
        self.lambda_no_obj = lambda_no_obj

    @torch.no_grad()
    def _hungarian_match(self, pred_times, pred_confs, gt_times, gt_mask):
        """Compute optimal bipartite matching for one sample.

        Args:
            pred_times: (n_queries,) predicted times in seconds
            pred_confs: (n_queries,) predicted confidences
            gt_times: (max_gt,) ground truth times (padded)
            gt_mask: (max_gt,) mask for valid GT entries

        Returns:
            matched_pred_idx: indices of matched predictions
            matched_gt_idx: indices of matched GT
        """
        pred_t = pred_times.cpu().numpy()
        pred_c = pred_confs.cpu().numpy()
        gt_t = gt_times.cpu().numpy()
        mask = gt_mask.cpu().numpy()

        n_pred = len(pred_t)
        n_gt = int(mask.sum())

        if n_gt == 0:
            return np.array([], dtype=int), np.array([], dtype=int)

        gt_valid = gt_t[:n_gt]

        # Cost matrix: (n_pred, n_gt)
        # Cost = |pred_time - gt_time| / 10.0 + lambda * (1 - pred_conf)
        time_cost = np.abs(pred_t[:, None] - gt_valid[None, :]) / 10.0
        conf_cost = self.lambda_no_obj * (1.0 - pred_c[:, None])
        cost_matrix = time_cost + conf_cost

        # Hungarian matching
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

        # Filter out matches with very high time cost (> 2 seconds)
        valid = np.abs(pred_t[pred_idx] - gt_valid[gt_idx]) < 2.0
        pred_idx = pred_idx[valid]
        gt_idx = gt_idx[valid]

        return pred_idx, gt_idx

    def forward(self, outputs, batch):
        """Compute total loss.

        Args:
            outputs: dict from E2EDischargeDetector.forward()
            batch: dict from E2EDataset collate

        Returns:
            total_loss: scalar tensor
            loss_dict: dict of individual loss components
        """
        pred_times = outputs['pred_times']   # (B, n_queries)
        pred_confs = outputs['pred_confs']   # (B, n_queries)
        pred_freq = outputs['pred_freq']     # (B,)
        lat_logit = outputs['lat_logit']     # (B,)

        gt_times = batch['gt_times']         # (B, 30)
        gt_mask = batch['gt_mask']           # (B, 30)
        gt_freq = batch['freq']              # (B,)
        gt_lat = batch['lat']                # (B,)

        B = pred_times.shape[0]
        device = pred_times.device

        total_time_loss = torch.tensor(0.0, device=device)
        total_conf_pos_loss = torch.tensor(0.0, device=device)
        total_conf_neg_loss = torch.tensor(0.0, device=device)
        n_matched_total = 0
        n_unmatched_total = 0

        for b in range(B):
            pred_idx, gt_idx = self._hungarian_match(
                pred_times[b], pred_confs[b], gt_times[b], gt_mask[b]
            )

            n_matched = len(pred_idx)
            n_queries = pred_times.shape[1]

            if n_matched > 0:
                pred_idx_t = torch.from_numpy(pred_idx).long().to(device)
                gt_idx_t = torch.from_numpy(gt_idx).long().to(device)

                # Time L1 loss on matched pairs
                matched_pred_t = pred_times[b][pred_idx_t]
                matched_gt_t = gt_times[b][gt_idx_t]
                total_time_loss = total_time_loss + F.l1_loss(
                    matched_pred_t, matched_gt_t, reduction='sum'
                )

                # Confidence BCE: matched -> 1
                total_conf_pos_loss = total_conf_pos_loss + F.binary_cross_entropy(
                    pred_confs[b][pred_idx_t],
                    torch.ones(n_matched, device=device),
                    reduction='sum'
                )
                n_matched_total += n_matched

            # Confidence BCE: unmatched -> 0
            all_idx = set(range(n_queries))
            unmatched_idx = sorted(all_idx - set(pred_idx.tolist()))
            if unmatched_idx:
                unmatched_idx_t = torch.tensor(unmatched_idx, dtype=torch.long,
                                                device=device)
                total_conf_neg_loss = total_conf_neg_loss + F.binary_cross_entropy(
                    pred_confs[b][unmatched_idx_t],
                    torch.zeros(len(unmatched_idx), device=device),
                    reduction='sum'
                )
                n_unmatched_total += len(unmatched_idx)

        # Normalize
        time_loss = total_time_loss / max(n_matched_total, 1)
        conf_pos_loss = total_conf_pos_loss / max(n_matched_total, 1)
        conf_neg_loss = total_conf_neg_loss / max(n_unmatched_total, 1)
        conf_loss = conf_pos_loss + conf_neg_loss

        # Frequency loss (Huber, only for samples with known frequency)
        freq_mask = gt_freq > 0  # -1 means unknown
        if freq_mask.any():
            freq_loss = F.huber_loss(
                pred_freq[freq_mask], gt_freq[freq_mask], reduction='mean', delta=0.5
            )
        else:
            freq_loss = torch.tensor(0.0, device=device)

        # Laterality loss (BCE, only for samples with known laterality)
        lat_mask = gt_lat >= 0  # -1 means unknown
        if lat_mask.any():
            # Convert lat_logit to probability via sigmoid
            lat_pred = torch.sigmoid(lat_logit[lat_mask] * 5.0)  # scale for sharper sigmoid
            lat_loss = F.binary_cross_entropy(
                lat_pred, gt_lat[lat_mask], reduction='mean'
            )
        else:
            lat_loss = torch.tensor(0.0, device=device)

        # Total loss
        total = (self.w_time * time_loss +
                 self.w_conf * conf_loss +
                 self.w_freq * freq_loss +
                 self.w_lat * lat_loss)

        loss_dict = {
            'total': total.item(),
            'time': time_loss.item(),
            'conf_pos': conf_pos_loss.item(),
            'conf_neg': conf_neg_loss.item(),
            'freq': freq_loss.item(),
            'lat': lat_loss.item(),
            'n_matched': n_matched_total,
        }

        return total, loss_dict
