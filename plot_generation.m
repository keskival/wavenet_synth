load("sound.mat")
load("image.mat")
subplot(1,1,1)
size(s)
seed_length = 1024 * 512;
s = s(:,seed_length:size(s)(2));
subplot(1, 2, 1)
plot(s)
soundsc(s, 48000)
subplot(1, 2, 2)
imagesc(flipud(i'))
wavwrite(s, 48000, "generated.wav")
