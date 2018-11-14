import numpy as np
import tensorflow as tf
import json
import pickle
import data_utils
import plotting
import model
import utils

from time import time
from mmd import median_pairwise_distance, mix_rbf_mmd2_and_ratio

tf.logging.set_verbosity(tf.logging.ERROR)

# --- get settings --- #
# parse command line arguments, or use defaults
parser = utils.rgan_options_parser()
settings = vars(parser.parse_args())
# if a settings file is specified, it overrides command line arguments/defaults
if settings['settings_file']: settings = utils.load_settings_from_file(settings)

# --- get data, split --- #
samples, pdf, labels = data_utils.get_samples_and_labels(settings)



# --- save settings, data --- #
print('Ready to run with settings:')
for (k, v) in settings.items(): print(v, '\t', k)
# add the settings to local environment
# WARNING: at this point a lot of variables appear
locals().update(settings)
json.dump(settings, open('./experiments/settings/' + identifier + '.txt', 'w'), indent=0)

if data == 'eICU_task':
    train_seqs = samples['train'].reshape(-1, 16, 4)
    vali_seqs = samples['vali'].reshape(-1, 16, 4)
    test_seqs = samples['test'].reshape(-1, 16, 4)
    train_targets = labels['train']
    vali_targets = labels['vali']
    test_targets = labels['test']

if not data == 'load':
    data_path = './experiments/data/' + identifier + '.data.npy'
    np.save(data_path, {'samples': samples, 'pdf': pdf, 'labels': labels})
    print('Saved training data to', data_path)

# --- build model --- #

Z, X, CG, CD, CS = model.create_placeholders(batch_size, seq_length, latent_dim,
                                             num_signals, cond_dim)

discriminator_vars = ['hidden_units_d', 'seq_length', 'cond_dim', 'batch_size', 'batch_mean']
discriminator_settings = dict((k, settings[k]) for k in discriminator_vars)
generator_vars = ['hidden_units_g', 'seq_length', 'batch_size',
                  'num_generated_features', 'cond_dim', 'learn_scale']
generator_settings = dict((k, settings[k]) for k in generator_vars)

CGAN = (cond_dim > 0)
if CGAN: assert not predict_labels

D_loss, G_loss = model.GAN_loss(Z, X, generator_settings, discriminator_settings,
                                kappa, CGAN, CG, CD, CS, wrong_labels=wrong_labels)
D_solver, G_solver, priv_accountant = model.GAN_solvers(D_loss, G_loss, learning_rate, batch_size,
                                                        total_examples=samples['train'].shape[0],
                                                        l2norm_bound=l2norm_bound,
                                                        batches_per_lot=batches_per_lot, sigma=dp_sigma, dp=dp)
G_sample = model.generator(Z, **generator_settings, reuse=True, c=CG)

# --- evaluation --- #

# frequency to do visualisations
vis_freq = 50
eval_freq = 50

# get heuristic bandwidth for mmd kernel from evaluation samples
heuristic_sigma_training = median_pairwise_distance(samples['vali'])
best_mmd2_so_far = 1000

# optimise sigma using that (that's t-hat)
batch_multiplier = 5000 // batch_size
eval_size = batch_multiplier * batch_size
eval_eval_size = int(0.2 * eval_size)
eval_real_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
eval_sample_PH = tf.placeholder(tf.float32, [eval_eval_size, seq_length, num_generated_features])
n_sigmas = 2
sigma = tf.get_variable(name='sigma', shape=n_sigmas, initializer=tf.constant_initializer(
    value=np.power(heuristic_sigma_training, np.linspace(-1, 3, num=n_sigmas))))
mmd2, that = mix_rbf_mmd2_and_ratio(eval_real_PH, eval_sample_PH, sigma)
with tf.variable_scope("SIGMA_optimizer"):
    sigma_solver = tf.train.RMSPropOptimizer(learning_rate=0.05).minimize(-that, var_list=[sigma])
    # sigma_solver = tf.train.AdamOptimizer().minimize(-that, var_list=[sigma])
    # sigma_solver = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(-that, var_list=[sigma])
sigma_opt_iter = 2000
sigma_opt_thresh = 0.001
sigma_opt_vars = [var for var in tf.global_variables() if 'SIGMA_optimizer' in var.name]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

vis_Z = model.sample_Z(batch_size, seq_length, latent_dim, use_time)
if CGAN:
    vis_C = model.sample_C(batch_size, cond_dim, max_val, one_hot)
    if 'eICU_task' in data:
        vis_C = labels['train'][np.random.choice(labels['train'].shape[0], batch_size, replace=False), :]
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
else:
    vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
    vis_C = None

vis_real_indices = np.random.choice(len(samples['vali']), size=6)
vis_real = np.float32(samples['vali'][vis_real_indices, :, :])
if not labels['vali'] is None:
    vis_real_labels = labels['vali'][vis_real_indices]
else:
    vis_real_labels = None

if 'eICU' in data:
    plotting.vis_eICU_patients_downsampled(vis_real, resample_rate_in_min,
                                           identifier=identifier + '_real', idx=0)
else:
    plotting.save_plot_sample(vis_real, 0, identifier + '_real', n_samples=6,
                              num_epochs=num_epochs)

trace = open('./experiments/traces/' + identifier + '.trace.txt', 'w')
trace.write('epoch time D_loss G_loss mmd2 that pdf real_pdf\n')

# --- train --- #
train_vars = ['batch_size', 'D_rounds', 'G_rounds', 'use_time', 'seq_length',
              'latent_dim', 'num_generated_features', 'cond_dim', 'max_val',
              'WGAN_clip', 'one_hot']
train_settings = dict((k, settings[k]) for k in train_vars)

t0 = time()
best_epoch = 0
print('epoch\ttime\tD_loss\tG_loss')
for epoch in range(num_epochs):
    D_loss_curr, G_loss_curr = model.train_epoch(epoch, samples['train'], labels['train'],
                                                 sess, Z, X, CG, CD, CS,
                                                 D_loss, G_loss,
                                                 D_solver, G_solver,
                                                 **train_settings)
    # -- eval -- #

    # visualise plots of generated samples, with/without labels
    if epoch % vis_freq == 0:
        if CGAN:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z, CG: vis_C})
        else:
            vis_sample = sess.run(G_sample, feed_dict={Z: vis_Z})
        plotting.visualise_at_epoch(vis_sample, data,
                                    predict_labels, one_hot, epoch, identifier, num_epochs,
                                    resample_rate_in_min, multivariate_mnist, seq_length, labels=vis_C)

    # compute mmd2 and, if available, prob density
    if epoch % eval_freq == 0:
        model.dump_parameters(identifier + '_' + str(epoch), sess)
        t = time() - t0
        print('%d\t%.2f\t%.4f\t%.4f' % (epoch, t, D_loss_curr, G_loss_curr))
        if 'eICU' in data:
            gen_samples = []
            labels_gen_samples = []
            print(int(len(train_seqs) / batch_size))
            for batch_idx in range(int(len(train_seqs) / batch_size)/10):
                X_mb, Y_mb = data_utils.get_batch(train_seqs, batch_size, batch_idx, train_targets)
                z_ = model.sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
                gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG: Y_mb})
                gen_samples.append(gen_samples_mb)
                labels_gen_samples.append(Y_mb)
                print(batch_idx)

            for batch_idx in range(int(len(vali_seqs) / batch_size)/10):
                X_mb, Y_mb = data_utils.get_batch(vali_seqs, vali_targets, batch_size, batch_idx)
                z_ = model.sample_Z(batch_size, seq_length, latent_dim, use_time=use_time)
                gen_samples_mb = sess.run(G_sample, feed_dict={Z: z_, CG: Y_mb})
                gen_samples.append(gen_samples_mb)
                labels_gen_samples.append(Y_mb)

            gen_samples = np.vstack(gen_samples)
            labels_gen_samples = np.vstack(labels_gen_samples)

            wd = './synthetic_eICU_datasets'
            with open(wd + '/samples_' + identifier + '_' + str(epoch) + '.pk', 'wb') as f:
                pickle.dump(file=f, obj=gen_samples)

            with open(wd + '/labels_' + identifier + '_' + str(epoch) + '.pk', 'wb') as f:
                pickle.dump(file=f, obj=labels_gen_samples)

            # save the model used to generate this dataset
            model.dump_parameters(identifier + '_' + str(epoch), sess)

    if shuffle:  # shuffle the training data
        perm = np.random.permutation(samples['train'].shape[0])
        samples['train'] = samples['train'][perm]
        if labels['train'] is not None:
            labels['train'] = labels['train'][perm]

model.dump_parameters(identifier + '_' + str(epoch), sess)