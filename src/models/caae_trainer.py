import numpy as np
import tensorflow as tf

from utils.schedules import linear_lambda


@tf.function
def _train_step(model, label, input_x, input_y, optimizer, current_lambda):
    with tf.GradientTape() as tape:
        output = model((label, input_x, input_y), training=True)
        z = model.z
        rec_loss = model.reconstruction(input_y, output)
        dom_loss, dom_pred = model.domain_loss(label, z, lambd=current_lambda)

        # Reconstruction loss is (batch, feat_dim); reduce to a scalar for optimization.
        rec_loss_scalar = tf.reduce_mean(rec_loss)
        loss = rec_loss_scalar + current_lambda * dom_loss

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss, rec_loss_scalar, dom_loss, dom_pred


@tf.function
def _eval_step(model, label, input_x, input_y, current_lambda):
    output = model((label, input_x, input_y), training=False)
    z = model.z
    rec_loss = tf.reduce_mean(model.reconstruction(input_y, output))
    dom_loss, dom_pred = model.domain_loss(label, z, lambd=current_lambda)
    loss = rec_loss + current_lambda * dom_loss
    return loss, rec_loss, dom_loss, dom_pred


def evaluate(model, dataset, current_lambda):
    epoch_loss = tf.keras.metrics.Mean()
    epoch_rec = tf.keras.metrics.Mean()
    epoch_dom = tf.keras.metrics.Mean()
    epoch_acc = tf.keras.metrics.BinaryAccuracy()

    for (label, input_x, input_y) in dataset:
        loss, rec, dom, dom_pred = _eval_step(model, label, input_x, input_y, current_lambda)
        epoch_loss.update_state(loss)
        epoch_rec.update_state(rec)
        epoch_dom.update_state(dom)
        epoch_acc.update_state(tf.reshape(label, (-1, 1)), dom_pred)

    return epoch_loss.result(), epoch_rec.result(), epoch_dom.result(), epoch_acc.result()


def train(
    model,
    train_dataset,
    test_dataset,
    epochs,
    optimizer,
    max_iterations,
    max_lambda,
    perform_evaluation=True,
    # optional CF evaluation
    cf_eval_every=5,
    x_data_test=None,
    y_data_0_eval=None,
    y_data_1_test=None,
    label_real_test=None,
    label_cf_test=None,
):
    """
    Training loop for the CAAE baseline.

    Returns:
      cf_eval_epochs, cf_mae_history
    """
    iteration = 0
    cf_mae_history = []
    cf_eval_epochs = []

    for epoch in range(int(epochs)):
        train_loss = tf.keras.metrics.Mean()
        train_rec = tf.keras.metrics.Mean()
        train_dom = tf.keras.metrics.Mean()
        train_acc = tf.keras.metrics.BinaryAccuracy()

        for (label, input_x, input_y) in train_dataset:
            current_lambda = linear_lambda(iteration, max_iterations, max_lambda=max_lambda)
            current_lambda = tf.convert_to_tensor(current_lambda, dtype=tf.float32)

            loss, rec, dom, dom_pred = _train_step(
                model, label, input_x, input_y, optimizer, current_lambda
            )

            train_loss.update_state(loss)
            train_rec.update_state(rec)
            train_dom.update_state(dom)
            train_acc.update_state(tf.reshape(label, (-1, 1)), dom_pred)

            iteration += 1

        if perform_evaluation:
            test_loss, test_rec, test_dom, test_acc = evaluate(model, test_dataset, current_lambda)
            print(
                f"Epoch {epoch+1} | "
                f"Train L={train_loss.result().numpy():.4f} (rec {train_rec.result().numpy():.4f}, dom {train_dom.result().numpy():.4f}, acc {train_acc.result().numpy():.3f}) | "
                f"Test L={test_loss.numpy():.4f} (rec {test_rec.numpy():.4f}, dom {test_dom.numpy():.4f}, acc {test_acc.numpy():.3f})"
            )
        else:
            print(f"Epoch {epoch+1}")

        # Optional CF evaluation
        do_cf_eval = (
            cf_eval_every is not None
            and cf_eval_every > 0
            and (epoch + 1) % cf_eval_every == 0
            and x_data_test is not None
            and y_data_0_eval is not None
            and y_data_1_test is not None
            and label_real_test is not None
            and label_cf_test is not None
        )

        if do_cf_eval:
            pred = np.array(
                model.cf_generation(
                    label_real=label_real_test,
                    label_cf=label_cf_test,
                    x=x_data_test,
                    y=y_data_1_test,
                )
            )
            mae_list = np.mean(np.abs(y_data_0_eval - pred), axis=1)
            mae = float(np.mean(mae_list))
            mbe = float(np.mean(y_data_0_eval - pred))
            var = float(np.var(mae_list))

            cf_mae_history.append(mae)
            cf_eval_epochs.append(epoch + 1)

            print(f"[CF Eval] Epoch {epoch+1} | MAE={mae:.4f} | Var={var:.4f} | MBE={mbe:.4f}")

    return cf_eval_epochs, cf_mae_history
