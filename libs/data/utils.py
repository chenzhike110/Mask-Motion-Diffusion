import torch

# padding to max length in one batch
def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

# padding t pose in one batch
def collate_t_pose(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)

def mld_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    notnone_batches.sort(key=lambda x: x[2], reverse=True)
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = {
        "pose_body":
        collate_tensors([torch.tensor(b[0]).float() for b in notnone_batches]),
        "text": [b[3] for b in notnone_batches],
        "length": torch.tensor([b[2] for b in notnone_batches], dtype=torch.int64),
        "pose_root":
        collate_tensors([torch.tensor(b[1]).float() for b in notnone_batches]),
        "tokens": [b[4] for b in notnone_batches],
    }
    return adapted_batch