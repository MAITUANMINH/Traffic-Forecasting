# Ve hinh do nhay r va tau tu ket qua results/S_*.csv, B_BigST.csv, C_AB*.csv (so lieu chep tu cac file do)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
r=[16,32,64]; mr=[1.4913,1.4885,1.4843]
ax[0].plot(r,mr,'o-',color='tab:blue',label='BigST gốc (τ = 0,2)')
ax[0].plot([16,64],[1.4758,1.4701],'^--',color='tab:green',label='BigST cải tiến A+B (τ = 0,25)')
ax[0].set_xscale('log',base=2); ax[0].set_xticks(r); ax[0].set_xticklabels(r)
ax[0].set_xlabel('Số chiều đặc trưng ngẫu nhiên r'); ax[0].set_ylabel('MAE trung bình 12 bước')
ax[0].set_title('(a) Ảnh hưởng của r'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)
t=[0.2,0.25,0.6,1.0]; mt=[1.4843,1.4837,1.4881,1.4885]
ax[1].plot(t,mt,'o-',color='tab:blue',label='BigST gốc (r = 64)')
ax[1].set_xlabel('Hệ số nhiệt độ τ'); ax[1].set_ylabel('MAE trung bình 12 bước')
ax[1].set_title('(b) Ảnh hưởng của τ'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
for a in ax: a.set_ylim(1.465,1.495)
plt.tight_layout(); plt.savefig('results/do_nhay_r_tau.png',dpi=200)
