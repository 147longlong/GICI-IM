import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

font1 = font_manager.FontProperties(family='Times New Roman', size=8)
font2 = font_manager.FontProperties(family='Times New Roman', size=10)


def standford_plot(
    PE,
    PL,
    AL,
    x_max=25,
    y_max=25,
    plotname='Stanford Plot',
    figsize=(16, 16),
    save_path=None,
    show=False,
):
    plt.figure(figsize=figsize)

    PE = np.nan_to_num(PE, nan=99)
    PL = np.nan_to_num(PL, nan=999)

    x  = np.linspace(0, x_max, 100)
    x1 = np.linspace(0, AL, 100)  
    x2 = np.linspace(AL, x_max, 100)

    import matplotlib.colors as colors    
    regions = {
        'I': lambda PE, PL: (PE <= PL) & (PL < AL),
        'II': lambda PE, PL: (PL < PE) & (PE < AL),
        'III': lambda PE, PL: (PE < AL) & (AL < PL ),
        'IV': lambda PE, PL: (AL < PE) & (PE < PL),
        'V': lambda PE, PL: (AL < PL) & (PL < PE),
        'VI': lambda PE, PL: (PL < AL) & (AL < PE),
    }

    counts = {region: np.sum(func(np.array(PE), np.array(PL))) for region, func in regions.items()}
    prob_I = counts['I'] / len(PL) * 100

    plt.plot(x, AL*np.ones(len(x)),'--', color="black", label='Alert Limit', linewidth=0.5)
    plt.plot(AL*np.ones(len(x)), x,'--', color="black", label='Alert Limit', linewidth=0.5)
    plt.plot(x, x,'--', color="black", label='PE = PL',linewidth=0.6)
    plt.fill_between(x1,x1,AL, color='white', alpha=0.5)
    plt.fill_between(x1,0,x1, color=[246/255,184/255,171/255])
    plt.fill_between(x2,0,AL, color=[237/255,99/255,89/255])
    plt.fill_between(x2,AL,x2, color=[241/255,172/255,47/255])
    plt.fill_between(x1,AL,y_max, color=[249/255,219/255,87/255])
    plt.fill_between(x2,x2,y_max, color=[249/255,219/255,87/255])
    plt.text(0.1, AL, 'Alert Limit(AL)', ha='left', va='bottom')
    plt.text(AL, 0.1, 'Alert Limit(AL)', ha='left', va='bottom', rotation=-90)

    I_x_center = 0+0.03
    I_y_center = np.mean((x1 + AL) / 2)
    plt.text(I_x_center, I_y_center, f'Nominal operation \n epochs:{counts["I"]}, {prob_I:.2f}% ', ha='left', va='center')
    II_x_center = AL - 0.1
    II_y_center = np.mean((0 + AL) / 4)
    plt.text(II_x_center, II_y_center, f'Misleading operation\n epochs:{counts["II"]} ', ha='right', va='center')
    III_IV_x_center = np.mean(x)
    III_IV_y_center = np.mean((AL + y_max) / 2) 
    plt.text(III_IV_x_center, III_IV_y_center, f'System unavailable \n epochs:{counts["III"] + counts["IV"]} ', ha='center', va='center')
    V_x_center = np.mean(x_max)
    V_y_center = np.mean(AL)
    plt.text(V_x_center, V_y_center, f'System unavailable \n & Misleading information \n epochs:{counts["V"]} ', ha='right', va='bottom')
    VI_x_center = x_max - 0.1
    VI_y_center = np.mean((AL) / 2)
    plt.text(VI_x_center, VI_y_center, f'Hazardous Operations \n epochs:{counts["VI"]} ', ha='right', va='bottom')

    norm = colors.LogNorm(vmin=1)
    counts, xedges, yedges = np.histogram2d(PE, PL, bins=[x_max, y_max])

    PE_idx = np.digitize(PE, xedges) - 1
    PL_idx = np.digitize(PL, yedges) - 1
    PE_idx = np.clip(PE_idx, 0, counts.shape[0] - 1)
    PL_idx = np.clip(PL_idx, 0, counts.shape[1] - 1)
    PE = np.array(PE)
    PL = np.array(PL)
    mask = np.where(counts[PE_idx, PL_idx] > 0)
    PE_masked = PE[mask]
    PL_masked = PL[mask]
    PE_masked = np.clip(PE_masked, None, x_max-0.02)
    PL_masked = np.clip(PL_masked, None, y_max-0.02)
    colors = counts[PE_idx, PL_idx][mask]
    plt.scatter(PE_masked, PL_masked,s=4, c=colors, cmap='winter', norm=norm)
    cbar = plt.colorbar(ax=plt.gca())
    cbar.set_label('Number of points per pixel')

    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.xlabel('Position Error(m)')
    plt.ylabel('Protection Level(m)')
    plt.title(plotname)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()