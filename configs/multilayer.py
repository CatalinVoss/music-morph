import time

class config():
    # env config
    render_train     = False
    render_test      = False
    overwrite_render = True
    record           = True
    high             = 255.
    epoch_time = int(time.time())
    # output config
    previous_chkpt = "results/multilayer1521005208" 
    output_path  = "results/multilayer" + str(epoch_time) + "/"
    model_output = output_path + "model.ckpt"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 10
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 25000
    log_freq          = 50
    eval_freq         = 250000
    record_freq       = 250000
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5000000
    batch_size         = 32
    buffer_size        = 1000000
    target_update_freq = 1000
    gamma              = 0.99
    learning_freq      = 4
    state_history      = 1
    skip_frame         = 4
    lr_begin           = 0.00025
    lr_end             = 0.00005
    lr_nsteps          = nsteps_train/2
    eps_begin          = 0.2
    eps_end            = 0.01
    eps_nsteps         = 1000000
    learning_start     = 500
