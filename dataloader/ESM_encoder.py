import esm, torch, os
import numpy as np
def ESM_encoder(dir, proname, seq, model, alphabet, batch_converter):
    data = [
        ("protein1", seq),
    ]
    if not os.path.exists(dir):
        os.mkdir(dir)
    if len(seq) > 1024:
        return torch.FloatTensor([], )
    llm_dir = os.path.join(dir, proname)
    if os.path.exists(llm_dir+'.npy'):
        try:
            llm = np.load(llm_dir+'.npy')
            if llm.shape[0] == len(seq):
                return torch.FloatTensor(np.load(llm_dir+'.npy'), )
        except:
            pass

    print('ESM predicting...')
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    np.save(llm_dir, token_representations[0,1:-1,:])
    print('ESM predicting over')
    return token_representations[0,1:-1,:]