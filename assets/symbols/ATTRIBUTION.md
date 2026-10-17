# Symbol artwork: attribution

The SVG files in this folder are IEC 60617 symbol artwork derived from the QElectroTech elements library
(https://qelectrotech.org/), by the QElectroTech team and contributors, licensed under the Creative Commons
Attribution 3.0 licence (CC BY 3.0, https://creativecommons.org/licenses/by/3.0/).

They were copied unchanged from synergycodes/ng-diagram-single-line-diagram
(https://github.com/synergycodes/ng-diagram-single-line-diagram, commit 449df49147f8095143351139ab9d3bdac0e4cc96, folder src/assets/symbols/),
whose NOTICE.md records the derivation: QElectroTech element sources converted to SVG symbol bodies on an 8 px grid.

Files: breaker.svg, disconnector.svg, ct.svg, ground.svg, protection-relay.svg, shunt-capacitor.svg, surge-arrester.svg.

How sld2quote uses them: `sld2quote/generate.py` draws them onto synthetic sheets at varying scale and line width,
adds the connecting lines itself, skips their text elements (the sheet draws that text so it is recorded in the ground
truth) and turns the breaker a quarter turn when it is drawn as a bus-section. The files themselves are not modified.

The fuse-switch, contactor, meter, motor and distribution-board shapes are drawn in `generate.py` to the same IEC 60617
conventions and do not come from this library.
