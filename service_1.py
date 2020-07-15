from flask import Flask,request,render_template,jsonify
import sys

from tqdm import tqdm

import torch

app=Flask(__name__,
          template_folder="templates",
          static_folder="static/css"
          )


model="cv/cv/lm_lstm_epoch49.66_1.2114.pt"
seed=123
gpuid=0
verbose=0
beamsize=10
sent=False

def gprint(s):
    s = s.replace('\n', '<n>')
    if verbose == 1:
        print(s, file=sys.stderr)

if gpuid >= 0 and torch.cuda.is_available():
    gprint('using CUDA on GPU {} ...'.format(gpuid))
    device = torch.device('cuda',gpuid)
else:
    gprint('Falling back on CPU mode')
    gpuid = -1  # overwrite user setting
    device = torch.device("cpu")

torch.manual_seed(seed)

# if not os.path.exists(model):
#     gprint("Error: File {} does not exist. Are you sure you didn't forget to prepend cv/ ?".format(model))
checkpoint = torch.load(model)
protos = checkpoint.protos
protos.rnn.to(device)
protos.rnn.eval() # put in eval mode so that dropout works properly

# initialize the vocabulary (and its inverted version)
vocab = checkpoint.vocab

#initialize the rnn state to all zeros
gprint('creating an lstm...')
current_state = checkpoint.protos.rnn.init_hidden(1, device)

def beam_search_decoder(s, k, progress_bar=True):
    with torch.no_grad():
        h, c = current_state
        h, c = h.clone().squeeze(), c.clone().squeeze()
        sequences = [[s[0], (h, c), 0.]]

        # iterate over each character
        progress = range(1, len(s))
        if progress_bar:
            progress = tqdm(progress, desc="num characters")
        for t in progress:
            all_candidates = []

            # treat the candidates in the beam as one batch
            inputs = []
            hs = []
            cs = []
            for i in range(len(sequences)):
                seq, (h, c), _ = sequences[i]
                inputs.append(vocab.get(seq[-1], vocab["<unk>"]))
                hs.append(h)
                cs.append(c)

            # input shape: 1 x beam_size
            inputs = torch.tensor(inputs, device=device).unsqueeze(0)
            # hidden shape: num_layers x beam_size x H
            hs = torch.stack(hs).transpose(0, 1).contiguous()
            cs = torch.stack(cs).transpose(0, 1).contiguous()

            # forward the candidates in beam
            next_scores, (next_hs, next_cs) = protos.rnn(inputs, (hs, cs))

            # expand each current candidate
            for i in range(len(sequences)):
                seq, _, score = sequences[i]
                next_chars = [s[t]]
                if s[t].upper() != s[t]:
                    next_chars.append(s[t].upper())
                for c in next_chars:
                    next_score = next_scores[0, i, vocab.get(c, vocab["<unk>"])]
                    # same hidden/cell state is re-used for next char, do clone here to be safe
                    next_h, next_c = next_hs[:, i].clone(), next_cs[:, i].clone()
                    candidate = [seq + c, (next_h, next_c), score + next_score]
                    all_candidates.append(candidate)

            # order all candidates by highest score
            ordered = sorted(all_candidates, key=lambda tup:-tup[2])
            # select k best
            sequences = ordered[:k]

        decoded = sequences[0][0].strip()
        return decoded

@app.route("/getTrueCase",methods=['GET'])
def get_true_case():
    input_text= request.args.get("input")
    if sent:
        gprint('performing sentence-level truecasing...')
        gprint('memory will be reset after every line')
        lines = input_text
        output_list = list()
        for line in tqdm(lines, desc="num lines"):
            # truecase line by line
            output_list.append(beam_search_decoder('\n' + line.rstrip(), beamsize))
        output = ""
        output = output.join(output_list)
        print(output)
        #return output
    else:
        gprint('performing document-level truecasing...')
        gprint('memory is carried over to the next line')
        lines = input_text
        # truecase whole text at once
        output = beam_search_decoder('\n' + lines.rstrip(), beamsize)
        print(output)
        #return (output)
    return jsonify(output)
@app.route("/")
def true_case():
    return render_template("truecase.html")

if __name__=='__main__':
    app.run(debug=True,port=5000)