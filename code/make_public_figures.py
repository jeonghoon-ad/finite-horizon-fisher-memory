#!/usr/bin/env python3
"""Render the seven manuscript figures from frozen public results.

Presentation only: the upstream plotting functions still select every row and
compute every plotted coordinate. This module changes colours and text layout.
The census axis label follows its K_plus input and the manuscript definition.

Run from the package root, after the result links in REPRODUCE.md:
    python code/make_public_figures.py --out reproduced/public_figures
"""
from __future__ import annotations
import argparse
import importlib.util
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
WHITE, INK, GREY = '#FFFFFF', '#1A2744', '#66717E'


def style_figure(fig, name):
    """White backgrounds, legible secondary ink, and explicit label placement."""
    colour_map = {to_rgba('#F2EDE4'): WHITE, to_rgba('#9C8F7A'): GREY}
    for artist in fig.findobj():
        for attr in ('facecolor', 'edgecolor', 'color', 'markerfacecolor', 'markeredgecolor'):
            get, set_ = getattr(artist, 'get_'+attr, None), getattr(artist, 'set_'+attr, None)
            if get is None or set_ is None:
                continue
            try:
                value = get()
                try:
                    rgba = to_rgba(value)
                    for old, new in colour_map.items():
                        if rgba[:3] == old[:3]:
                            set_((*to_rgba(new)[:3], rgba[3]))
                except (ValueError, TypeError):
                    if hasattr(value, 'shape') and value.ndim == 2 and value.shape[1] == 4:
                        import numpy as np
                        replacement = value.copy()
                        for old, new in colour_map.items():
                            mask = np.all(np.isclose(value[:, :3], old[:3]), axis=1)
                            replacement[mask, :3] = to_rgba(new)[:3]
                        set_(replacement)
            except (ValueError, TypeError, AttributeError):
                pass
    for ax in fig.axes:
        for text in ax.texts:
            arrow = getattr(text, 'arrow_patch', None)
            if arrow is not None and to_rgba(arrow.get_edgecolor())[:3] == to_rgba('#9C8F7A')[:3]:
                arrow.set_color(GREY)
    fig.set_facecolor(WHITE)
    for ax in fig.axes:
        ax.set_facecolor(WHITE)
    if name == 'fig1_concept':
        for t in fig.axes[0].texts:
            if t.get_text() == 'only the closure of $K$ changes':
                t.set_text('only the closure of $K$\nchanges')
                t.set_va('center')
    elif name == 'fig2_elliptic':
        a,b = fig.axes
        fig.set_size_inches(9.4,4.4)
        a.set_position([.06,.19,.35,.73]); b.set_position([.56,.19,.42,.73])
        b.legend(fontsize=9,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.17))
        for t in b.texts:
            if t.get_text().startswith('dots:'):
                t.set_position((35,1.50))
            if t.get_text().startswith('normal carrier:'):
                t.set_fontsize(8.5)
        # Keep both labels comfortably inside the left panel's lateral margins.
        for t in a.texts:
            if '\\lambda_{min}' in t.get_text(): t.set_position((1.05,.25)); t.set_fontsize(10)
    elif name == 'fig3_census':
        ax=fig.axes[0]
        ax.set_xlabel('forward finite-window gain $\\sigma_{4096}=\\max_{0\\leq j\\leq4096}\\|W^j\\|_2$\nmedians and min–max ranges over 8 draws', fontsize=10)
        ax.set_ylabel('best-direction Fisher total $\\lambda_{max}(M_{2048})$',fontsize=11)
        ax.set_title('Directional concentration and transient gain',fontsize=12)
        fig.subplots_adjust(left=.12,right=.98,bottom=.23,top=.9)
    elif name == 'fig4_factorial':
        ax=fig.axes[0]
        ax.set_ylabel('store-only Fisher information\noldest input $J^{(s)}_n(t=0)$',fontsize=11)
    elif name == 'figS7a_calibration':
        fig.set_size_inches(7.2,5.0)
        ax,axr=fig.axes
        ax.set_ylabel('empirical accuracy\n(recalibrated readout)',fontsize=11)
        ax.set_title('Fisher information and delayed accuracy',fontsize=12,pad=12)
        for t in ax.texts:
            if t.get_text().startswith('plotted'):
                t.set_text(t.get_text().replace('    least-squares','\nleast-squares'))
                t.set_fontsize(10)
            elif t.get_text().startswith('v_bottom'):
                t.set_fontsize(9.5)
        ax.legend(frameon=False,fontsize=10,loc='upper left',bbox_to_anchor=(.02,.86),markerscale=1.8)
        axr.set_ylabel('empirical\n$-$ predicted',fontsize=11)
        for t in axr.texts: t.set_fontsize(10)
    elif name == 'figS7b_objective_crossing':
        fig.set_size_inches(9.3,4.7)
        for ax in fig.axes:
            ax.set_xlabel('other readout objective\n(same carrier)',fontsize=11)
            for t in ax.texts: t.set_fontsize(10)
        fig.axes[0].legend(frameon=False,fontsize=10,ncol=3,loc='upper left',bbox_to_anchor=(0,-.24))
        fig.suptitle('Each readout dominates on its own target time',fontsize=12,y=1.0)
    elif name == 'figS7c_isolation_transport':
        axl,axr=fig.axes
        fig.set_size_inches(10.2,5.0)
        axl.set_position([.08,.25,.44,.63]);axr.set_position([.64,.25,.34,.63])
        axl.set_title('Isolated storage preserves accuracy;\nthe frozen readout ages',fontsize=11,pad=12)
        axr.set_title('Approximate isolation:\nretention and its lower bound',fontsize=11,pad=12)
        handles=axl.get_legend().legend_handles
        labels=[t.get_text() for t in axl.get_legend().get_texts()]
        axl.legend(handles,labels,frameon=False,fontsize=9.5,ncol=2,loc='upper left',bbox_to_anchor=(0,-.21))
        for t in axl.texts:
            if 'readout trained here' in t.get_text():
                t.set_y(.62)
        for t in axr.texts:
            t.set_text(t.get_text().replace('smallest bound_margin in the file:','smallest bound margin:').replace('  (floating-point scale)', '\n(floating-point scale)'))
            t.set_fontsize(9)
        axr.legend(frameon=False,fontsize=9,loc='lower right')
    # Nothing outside an axes should be clipped at its frame.
    for t in fig.findobj(match=Text):
        t.set_clip_on(False)


def save_figure(fig, out, name):
    style_figure(fig,name)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf'):
        kwargs={'dpi':300} if suffix=='png' else {'metadata':{'CreationDate':None,'ModDate':None}}
        fig.savefig(out/f'{name}.{suffix}',facecolor=WHITE,bbox_inches='tight',pad_inches=.14,**kwargs)
    plt.close(fig)


def load_overview():
    source=ROOT/'vendor/PREPRINT_PUBLIC_RELEASE_v1.2_20260917/scripts/make_preprint_figures_20260903.py'
    spec=importlib.util.spec_from_file_location('public_overview_upstream',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'reproduced/public_figures')
    args=parser.parse_args()
    upstream=load_overview()
    upstream.save=lambda fig,name:save_figure(fig,args.out,name)
    for name in ('fig1_concept','fig2_elliptic','fig3_census','fig4_factorial'):
        getattr(upstream,name)()
    import make_section7_figures as s7
    for name in ('fig_s7a_calibration','fig_s7b_objective_crossing','fig_s7c_isolation_transport'):
        if not getattr(s7,name)(ROOT/'results',args.out):
            raise RuntimeError('Required figure inputs missing: '+name)


if __name__=='__main__':
    main()
