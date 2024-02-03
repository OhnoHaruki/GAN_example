import os
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pylab
import numpy as np

from GAN_model import Discriminator
from GAN_model import Generator

def denorm(x):
    out = (x + 1) /2
    return out.clamp(0,1)

if __name__ == '__main__':
    
    #各変数の定義
    latent_size = 64
    hidden_size = 256
    image_size = 784
    num_epochs = 300
    batch_size = 32
    sample_dir = 'samples'
    save_dir = 'save'
    
    #GPUの指定と最適化
    device = torch.device("cuda:0")
    cudnn.benchmark = True# cuDNN最適化


    # ディレクトリの作成
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 画像の前処理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5,), # MNISTは1チャンエルのみ (PyTorch 1.1)
                                    std=(0.5,))])

    # MNISTデータセット
    mnist = torchvision.datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size, 
                                          shuffle=True)

    # Discriminator
    D = Discriminator(image_size, hidden_size).to(device)
    
    # Generator 
    G = Generator(image_size, latent_size, hidden_size).to(device)


    # バイナリ交差エントロピー損失と最適化アルゴリズム(Adam)
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


    # 保存する統計データ
    d_losses = np.zeros(num_epochs)
    g_losses = np.zeros(num_epochs)
    real_scores = np.zeros(num_epochs)
    fake_scores = np.zeros(num_epochs)

    fixed_noise = torch.randn(batch_size, latent_size).to(device)

    # Start training
    preview_count = 1
    total_step = len(data_loader)
    for epoch in range(num_epochs):
        for i, (images, _) in enumerate(data_loader):
            images = images.view(batch_size, -1).to(device)
            # BCE損失の入力として後に使用するラベルを作成する
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ================================================================== #
            #                     識別器の訓練
            # ================================================================== #

            # 本物の画像を利用して計算される BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
            #  real_labels == 1であるため、y = 1になって、損失の第二項は常に0になる。
            outputs = D(images)
            d_loss_real = criterion(outputs, real_labels)
            real_score = outputs
        
        
            # BCE損失は偽物画像を用いて計算される。
            # 損失の第１項目がここでは常に0であるため、fake_labels = 0
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
            d_loss_fake = criterion(outputs, fake_labels)
            fake_score = outputs
        
            # 逆伝播と最適化
            # Dが十分に訓練されていれば更新しない
            d_loss = d_loss_real + d_loss_fake
            
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            d_loss.backward()
            d_optimizer.step()
            # ================================================================== #
            #                        生成期の訓練                         #
            # ================================================================== #

            # 偽の画像を用いて損失を計算
            z = torch.randn(batch_size, latent_size).to(device)
            fake_images = G(z)
            outputs = D(fake_images)
        
            # Gは
            # log(D(G(z))を最大化し、log(1-D(G(z)))を最小化するような訓練をする。
            #  説明は→https://arxiv.org/pdf/1406.2661.pdf
            g_loss = criterion(outputs, real_labels)
        
            # 逆伝播と最適化
            # Gが十分に訓練されていれば更新しない。
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            g_loss.backward()
            g_optimizer.step()
            # =================================================================== #
            #                          統計情報の更新                          #
            # =================================================================== #
            d_losses[epoch] = d_losses[epoch] * (i/(i+1.)) + d_loss.item() * (1./(i+1.))
            g_losses[epoch] = g_losses[epoch] * (i/(i+1.)) + g_loss.item() * (1./(i+1.))
            real_scores[epoch] = real_scores[epoch] * (i/(i+1.)) + real_score.mean().item() * (1./(i+1.))
            fake_scores[epoch] = fake_scores[epoch] * (i/(i+1.)) + fake_score.mean().item() * (1./(i+1.))
        
            if (i+1) % 200 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                    .format(epoch, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                            real_score.mean().item(), fake_score.mean().item()))

            # 作成画像の保存
            if (i+1) % 200 == 0:
                with torch.no_grad():
                    sample_images = G(fixed_noise).detach().cpu()
                    sample_images = sample_images.view(sample_images.size(0), 1, 28, 28)
                    save_image(denorm(sample_images.data), os.path.join(sample_dir, 'fake_images-{:08}.png'.format(preview_count)))
                    preview_count = preview_count + 1

        # 本物画像の保存
        if (epoch + 1) == 1:
            images = images.view(images.size(0), 1, 28, 28)
            save_image(denorm(images.data), os.path.join(sample_dir, 'real_images.png'))


        # 統計情報を保存してプロット
        np.save(os.path.join(save_dir, 'd_losses.npy'), d_losses)
        np.save(os.path.join(save_dir, 'g_losses.npy'), g_losses)
        np.save(os.path.join(save_dir, 'fake_scores.npy'), fake_scores)
        np.save(os.path.join(save_dir, 'real_scores.npy'), real_scores)
    
        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        plt.plot(range(1, num_epochs + 1), d_losses, label='d loss')
        plt.plot(range(1, num_epochs + 1), g_losses, label='g loss')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'loss.pdf'))
        plt.close()

        plt.figure()
        pylab.xlim(0, num_epochs + 1)
        pylab.ylim(0, 1)
        plt.plot(range(1, num_epochs + 1), fake_scores, label='fake score')
        plt.plot(range(1, num_epochs + 1), real_scores, label='real score')    
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'accuracy.pdf'))
        plt.close()

        # その時点でのモデルをcheckpointsファイルに保存
        # checkpointsファイルとは→モデルの重み、バイアス、オプティマイザーの状態など、モデルのトレーニングに必要な情報が含まれる
        if (epoch + 1) % 50 == 0:
            torch.save(G.state_dict(), os.path.join(save_dir, 'G--{}.ckpt'.format(epoch+1)))
            torch.save(D.state_dict(), os.path.join(save_dir, 'D--{}.ckpt'.format(epoch+1)))

    # 最終的なモデルをcheckpointファイルに保存する 
    torch.save(G.state_dict(), 'G.ckpt')
    torch.save(D.state_dict(), 'D.ckpt')