load("i.mat")
load("s.mat")
subplot(1,2,1)
imagesc(flipud(i'))
subplot(1,2,2)
s = s(:,520000:size(s)(2));
plot(s)
soundsc(repmat(s, [1, 1]), 48000)
wavwrite(s, 48000, "generated.wav")
