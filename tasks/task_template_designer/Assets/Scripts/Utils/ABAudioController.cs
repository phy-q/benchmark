// SCIENCE BIRDS: A clone version of the Angry Birds game used for 
// research purposes
// 
// Copyright (C) 2016 - Lucas N. Ferreira - lucasnfe@gmail.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>
//

using UnityEngine;
using System.Collections;
using System.Collections.Generic;

[RequireComponent(typeof(AudioSource))]
public class ABAudioController : ABSingleton<ABAudioController> {
	
	private List<AudioSource> _independentAudioSources;

	void Awake() 
	{
		_independentAudioSources = new List<AudioSource>();
	}

	void Update()
	{
		for(int i = 0; i < _independentAudioSources.Count; i++)
		{
			if(!_independentAudioSources[i].isPlaying) {

				Destroy(_independentAudioSources[i].gameObject);
				_independentAudioSources.Remove(_independentAudioSources[i]);
			}
		}
	}

	public void PlayMusic(AudioClip music, bool loop = true)
	{
		gameObject.GetComponent<AudioSource>().clip = music;
		gameObject.GetComponent<AudioSource>().loop = loop;
		gameObject.GetComponent<AudioSource>().Play();
	}

	public bool IsPlayingMusic(AudioClip music) 
	{
		return gameObject.GetComponent<AudioSource>().isPlaying && 
			   gameObject.GetComponent<AudioSource>().clip == music;
	}

	public void StopMusic()
	{
		gameObject.GetComponent<AudioSource>().Stop();
	}

	public void PlayIndependentSFX(AudioClip music, bool loop = false)
	{
		GameObject independentSFX = new GameObject();
		independentSFX.transform.parent = transform;
		AudioSource indAudioSource = independentSFX.AddComponent<AudioSource>();
		indAudioSource.clip = music;
		indAudioSource.loop = loop;
		indAudioSource.Play();

		_independentAudioSources.Add (indAudioSource);
	}
}
