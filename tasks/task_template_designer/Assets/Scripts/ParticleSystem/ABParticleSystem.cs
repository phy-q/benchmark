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

﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ABParticleSystem : MonoBehaviour {

	public int		  _minVel, _maxVel;
	public int 		  _minAngle, _maxAngle;
	public float	  _minMass, _maxMass;
	public float 	  _minLifetime, _maxLifetime;
	public float 	  _systemLifetime;
	public float 	  _shootingRate;

	public bool 	  _addNoise;
	public bool 	  _applyGravity;
	public bool 	  _shootParticles;

	public GameObject[] _particleSprites;

	private float 	  _shootingTimer;
	private float 	  _selfDestructionTimer;

	private ABObjectPool<ABParticle> _particles;

	private void Start () {
	
		_particles = new ABObjectPool<ABParticle>(ABConstants.BLOCK_PARTCICLE_PER_SYSTEM, _particleSprites, InitParticle);
	}
	
	private void Update () {

		if (_shootParticles) {

			if (_systemLifetime > 0) {

				_selfDestructionTimer += Time.deltaTime;
				if (_selfDestructionTimer >= _systemLifetime) {

					_shootParticles = false;
				}
			}

			_shootingTimer += Time.deltaTime;

			if (_shootingTimer >= _shootingRate) {

				ShootParticle ();
				_shootingTimer = 0f;
			}
		}
	}

	private void InitParticle(ABParticle particle) {

		particle.Create (this);
	}

	public ABParticle ShootParticle() {

		ABParticle inactiveParticle = _particles.GetFreeObject();

		if (inactiveParticle != null) {

			float mass = Random.Range (_minMass, _maxMass);
			float randVel = Random.Range (_minVel, _maxVel);
			float randAng = Random.Range (_minAngle, _maxAngle) * Mathf.Deg2Rad;
			float lifetime = Random.Range (_minLifetime, _maxLifetime);

			Vector2 velocity = new Vector2 (Mathf.Cos(randAng), Mathf.Sin(randAng)) * randVel;

			inactiveParticle.mass = mass;
			inactiveParticle.Shoot (velocity, lifetime, _applyGravity, _addNoise);
		}

		return inactiveParticle;
	}

	public void SetParticlesParent(Transform parent) {

		_particles.SetObjectsParent (parent);
	}

	public List<ABParticle> GetUsedParticles() {

		return _particles.GetUsedObjects ();
	}

	public void KillAllParticles() {

		_particles.FreeAllObjects ();
	}

	public void KillParticle(ABParticle particle) {

		_particles.SetFreeObject (particle.gameObject);
	}
}
