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

public class ABBBirdBlue : ABBird
{

    public int _childrenAmount = 3;
    public float _angleBetweenBirds = 5f;

    private ABBBirdBlue[] _childrenBirds;

    void InitSpecialPower()
    {

        _childrenBirds = new ABBBirdBlue[_childrenAmount];

        for (int i = 0; i < _childrenAmount; i++)
        {

            GameObject obj = (GameObject)Instantiate(ABWorldAssets.BIRDS["BirdBlue"], transform.parent);
            obj.name = "BlueBird_Child" + i;
            obj.transform.position = transform.position;

            // Set layer to avoid collision.
            obj.layer = LayerMask.NameToLayer("Ghostspot");

            _childrenBirds[i] = obj.GetComponent<ABBBirdBlue>();
            _childrenBirds[i]._spriteRenderer.color = Color.clear;
        }
    }

    void SpecialAttack()
    {

        ABGameWorld.Instance.KillBird(this);

        float currentBirdAngle = _angleBetweenBirds * 0.5f * _childrenBirds.Length;
        Vector2 direction = Quaternion.AngleAxis(currentBirdAngle, Vector3.forward) * _rigidBody.velocity.normalized;

        foreach (ABBBirdBlue child in _childrenBirds)
        {

            // Set layer to allow collision.
            child.gameObject.layer = LayerMask.NameToLayer("Birds");

            // Reset Rigidbody Constraints.
            child._rigidBody.constraints = RigidbodyConstraints2D.None;
            child.IsFlying = IsFlying;
            child.OutOfSlingShot = OutOfSlingShot;
            child.JumpToSlingshot = JumpToSlingshot;
            child.transform.position = transform.position;

            child.CancelInvoke("IdleJump");

            child._spriteRenderer.color = Color.white;
            child._rigidBody.gravityScale = _rigidBody.gravityScale;
            child._rigidBody.velocity = direction * _rigidBody.velocity.magnitude;

            if (child._trailParticles)
                child._trailParticles._shootParticles = true;

            direction = Quaternion.AngleAxis(-_angleBetweenBirds, Vector3.forward) * direction.normalized;
        }

        _rigidBody.velocity = Vector2.zero;
        _collider.enabled = false;
        _spriteRenderer.color = Color.clear;
        _trailParticles._shootParticles = false;
    }

    protected override void Update()
    {

        base.Update();

        int aliveBirds = 0;

        if (_childrenBirds == null)
            return;

        foreach (ABBBirdBlue bird in _childrenBirds)
        {
            if (bird != null)
                aliveBirds++;
        }

        if (aliveBirds == 0)
            Die(false);
    }
}
