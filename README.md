# ECG Viz Libray
Author: Vineet Tiruvadi 2019

A fun library I made during my IM rotation at Grady/Emory. It's meant to help visualize the learning process with ECGs, particularly with an emphasis on seeing the traditional ECG timetraces in a more modern, dynamical systems perspective. Maybe this will rely on other control theory-related repos I've got setup...

## Phase portrait
ECGs are multiple measurements of a single underlying process: the heart beating.
As it beats, it exhibits *patterns* that cardiologists have spent a lot of time mapping in detail.
Those details, while important, can likely be simplified greatly by looking at the data in a different way.
The engineering and physics fields give us a great way to view them: the phase space.
In this space, we can see how variables relate to each other and how they change with respect to each other.
With a very simple reframing of the data, we can see patterns emerge much more obviously.

An example of the phase space between channel V1 and V2 is displayed below

![Example phase portrait](imgs/ECG_phase.png)
